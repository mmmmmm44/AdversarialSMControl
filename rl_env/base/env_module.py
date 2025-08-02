"""
Abstract base class for Smart Meter RL environment implementations.

This module defines the common interface and shared functionality for different
types of Smart Meter RL environments used in the project.
"""

from decimal import Decimal
import numpy as np
import gymnasium as gym
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Optional, Dict, Tuple
import struct
import json
from pathlib import Path

from model.H_network.base.h_network_module import HNetworkRLModuleBase
from rl_env.training_mode import TrainingMode
from rl_env.render_window import RenderWindowControl, RenderWindowMessageType
from rl_env.data_loader import BaseSmartMeterDataLoader
from rl_env.tariff import TIME_OF_USE_PRICES, STANDING_CHARGE
from utils import print_log


class SmartMeterEnvironmentBase(gym.Env, ABC):
    """
    Abstract base class for Smart Meter RL environments.
    
    Contains the common functionality shared between continuous and discrete
    Smart Meter RL environment implementations, using the Template Method Pattern
    for action-space specific operations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, smart_meter_data_loader: BaseSmartMeterDataLoader, h_network_rl_module: HNetworkRLModuleBase, mode: TrainingMode, log_folder: Path, rb_config: Optional[dict] = None, reward_lambda: float = 0.5, render_mode=None, render_host='127.0.0.1', render_port=50007):
        '''
        Initializes the SmartMeterEnvironmentBase.
        
        Args:
            smart_meter_data_loader (BaseSmartMeterDataLoader): Data loader for smart meter data (supports both simple and curriculum loaders).
            h_network_rl_module: H-network RL module for providing per-step privacy-related signals.
            mode (TrainingMode): The mode of the environment (train, validate, test).
            log_folder (Path): Folder to save logs and results.
            rb_config (Optional[dict]): Configuration for the rechargeable battery. If None, default values are used.
            reward_lambda (float): Weighting factor for the reward function, between 0 and 1.
            render_mode (Optional[str]): Render mode for visualization. If 'human', it connects to a render server.
            render_host (str): Host address of the render server.
            render_port (int): Port number of the render server.
        '''
        super(SmartMeterEnvironmentBase, self).__init__()

        # TCP client for real-time rendering
        self.render_host = render_host
        self.render_port = render_port
        self.render_mode = render_mode
        self.render_client_socket = None
        self.render_connected = False

        # RL environment attributes
        self.smart_meter_data_loader = smart_meter_data_loader
        self.h_network_rl_module = h_network_rl_module
        self.mode = mode
        
        # Check if data loader supports curriculum learning
        self.curriculum_info = getattr(self.smart_meter_data_loader, 'curriculum_info', {'curriculum_enabled': False})
        
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Using data loader: {type(self.smart_meter_data_loader).__name__}")
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Curriculum enabled: {self.curriculum_info.get('curriculum_enabled', False)}")
        
        if self.curriculum_info.get('curriculum_enabled', False):
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode lengths available: {list(self.curriculum_info.get('episodes_by_length', {}).keys())} days")
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Total episodes: {self.curriculum_info.get('total_episodes', 0)}")

        self.selected_idx = 0   # selected index of the segment to use
        
        # Initialize environment-specific components (action/observation spaces, battery, episode)
        self._init_environment_specifics(rb_config)

        # Reward function parameters
        self.reward_lambda = reward_lambda
        
        # H-network inference buffer for episode
        self.h_network_episode_inference_buffer = []
        
        # Create log folder if it doesn't exist
        self.log_folder = log_folder
        if not self.log_folder.exists():
            self.log_folder.mkdir(parents=True)
        
        # Episode data tracking
        self.episodes_rewards = []
        self.per_episode_rewards = []
        
        # Training timestep tracking (set by external training loop)
        self.training_timestep = 0
        
        # TCP client for real-time rendering
        if self.render_mode == "human":
            self._connect_to_render_server()
        else:
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}]] Render mode set to '{self.render_mode}'. Render server at {self.render_host}:{self.render_port}. render_connected: {self.render_connected}. render_client_socket: {self.render_client_socket}")

    def _connect_to_render_server(self):
        """Connect to the render server for real-time visualization."""
        try:
            import socket
            self.render_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.render_client_socket.connect((self.render_host, self.render_port))
            self.render_connected = True
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}]] Render mode set to '{self.render_mode}'. Render server at {self.render_host}:{self.render_port}. render_connected: {self.render_connected}. render_client_socket: {self.render_client_socket}")
        except Exception as e:
            self.render_connected = False
            self.render_client_socket = None
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Could not connect to render server: {e}")

    @abstractmethod
    def _init_environment_specifics(self, rb_config: Optional[dict]):
        """Initialize environment-specific components (action/observation spaces, battery, episode)."""
        pass

    @abstractmethod
    def _get_obs(self) -> Dict:
        """Get observation for the current step - implemented by concrete classes."""
        pass
    
    @abstractmethod
    def _get_info(self, obs: Dict, **kwargs) -> Dict:
        """Get additional information for the current step - implemented by concrete classes."""
        pass

    @abstractmethod
    def reset(self, episode_idx: int = None, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Start a new episode - core environment logic."""
        super(SmartMeterEnvironmentBase, self).reset(seed=seed, options=options)

    @abstractmethod  
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one timestep - core environment dynamics."""
        pass

    # --------------------
    # Functions for computing privacy and cost signals
    # --------------------

    @abstractmethod
    def _f_signal(self, y_t, z_t, y_t_plus_1) -> float:
        """Calculate the f_signal (privacy signal) for the reward function - implemented by concrete classes."""
        pass

    @abstractmethod
    def _g_signal(self, s_t_datetime: datetime, s_t_plus_1_datetime: datetime, power_kw: float) -> float:
        """
        Calculate the g_signal (cost signal) for the reward function - Implemented by concrete classes.
        """
        pass

    # Tariff
    # constants for time-of-use pricing model
    @staticmethod
    def _get_time_of_use_prices() -> Dict[Tuple[time, time], float]:
        """
        Get the time-of-use pricing model.
        
        Returns:
            Dictionary mapping time intervals to prices in GBP per kWh.
        """
        return TIME_OF_USE_PRICES

    @staticmethod
    def _get_standing_charge_per_day() -> float:
        """
        Get the standing charge per day in GBP.
        
        Returns:
            Standing charge per day.
        """
        return STANDING_CHARGE

    def _get_weighted_electricity_cost(s_t_datetime: datetime, s_t_plus_1_datetime: datetime) -> float:
        """
        Calculate the weighted electricity cost based on a time-of-use pricing model. This is a helper function for computing g_signal.
        Args:
            s_t_datetime (datetime): Current timestamp.
            s_t_plus_1_datetime (datetime): Next timestamp.
        Returns:
            float: Weighted electricity cost, across the time period. (i.e. delta_t * avg_price_per_kWh)
        """

        # compute the weighted electricity cost
        total_cost = 0.0
        for start_time, end_time in SmartMeterEnvironmentBase._get_time_of_use_prices().keys():
            if s_t_datetime.time() <= end_time and s_t_plus_1_datetime.time() >= start_time:
                # Calculate the overlap between the time intervals
                overlap_start = max(s_t_datetime, datetime.combine(s_t_datetime.date(), start_time))
                overlap_end = min(s_t_plus_1_datetime, datetime.combine(s_t_plus_1_datetime.date(), end_time))
                
                if overlap_start < overlap_end:
                    overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600.0  # Convert to hours
                    price_per_kwh = SmartMeterEnvironmentBase._get_time_of_use_prices()[(start_time, end_time)]
                    total_cost += overlap_duration * price_per_kwh

        return total_cost

    # --------------------
    # Functions for rendering and visualization
    # --------------------

    def render(self):
        """Render functionality - shared across implementations.
        
        This is intended to be called after a env.step() is called.
        """

        _current_step = self.episode.get_current_step() - 1  # get the current step (0-indexed)

        if self.render_mode == "human" and self.render_connected:
            try:
                # Get current state information
                if hasattr(self, 'episode') and _current_step < len(self.episode.df):
                    render_data = self._prepare_render_data(_current_step)
                    self._send_json_message({
                        "type": RenderWindowMessageType.DATA.name.lower(),
                        "payload": render_data
                    })
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Error during rendering: {e}")

    def _prepare_render_data(self, current_step: int) -> Dict:
        """Prepare data for rendering."""
        return {
            "timestamp": self.episode.df.iloc[current_step]['datetime'].isoformat(),
            "user_load": float(self.episode.df.iloc[current_step]['aggregate']),
            "grid_load": float(self.episode.df.iloc[current_step].get('grid_load', 0)),
            "battery_soc": float(self.battery.get_normalized_state_of_charge()),
        }
    
    def _send_control_message(self, command, payload=None):
        """Send a control message to the render window."""
        message = {
            'type': RenderWindowMessageType.CONTROL.name.lower(),
            'command': command,
        }
        if payload is not None:
            message['payload'] = payload
        self._send_json_message(message)

    def reset_render_window(self):
        """Reset the render window."""
        if self.render_mode == "human" and self.render_connected:
            try:
                self._send_control_message(RenderWindowControl.RESET.name)
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Error resetting render window: {e}")

    def save_render_graph(self, kwargs=None):
        """
        Save the current render graph.
        
        Args:
            kwargs: Additional arguments for saving the graph, which are valid arguments for plt.savefig().
        """
        if self.render_mode == "human" and self.render_connected:
            try:
                self._send_control_message(RenderWindowControl.SAVE_GRAPH.name, kwargs)
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Error saving render graph: {e}")

    def send_env_info(self):
        """Send environment info to render window."""
        if self.render_mode == "human" and self.render_connected:
            try:
                env_info = {
                    'selected_idx': self.selected_idx,
                }
                self._send_control_message(RenderWindowControl.RECEIVE_ENV_INFO.name, {'env_info': env_info})
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Error sending env info to render window: {e}")

    def _send_json_message(self, message: dict):
        """ Send a JSON message to the render server.
        Args:
            message (dict): The message to send, should contain 'type' and 'payload'.
        """
        if self.render_connected and self.render_client_socket:
            try:
                json_data = json.dumps(message).encode('utf-8')
                msg_len = struct.pack('>I', len(json_data))     # include a 4-byte length prefix to tell the render server how long the message is (for TCP stream)
                self.render_client_socket.sendall(msg_len + json_data)
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Failed to send message to render server: {e}")

    def _send_control_message(self, command, payload=None):
        """Send a control message to the render window."""
        message = {
            'type': RenderWindowMessageType.CONTROL.name.lower(),
            'command': command,
        }
        if payload is not None:
            message['payload'] = payload
        self._send_json_message(message)

    def set_training_timestep(self, timestep: int):
        """Set the current training timestep."""
        self.training_timestep = timestep
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Training timestep set to {timestep}")


    # --------------------
    # Functions related to episode component
    # --------------------
    def get_curriculum_probabilities(self):
        """
        Get current curriculum probabilities if using curriculum loader.
        Returns:
            dict: Dictionary mapping episode length to probability, or empty dict if not using curriculum
        """
        if hasattr(self.smart_meter_data_loader, 'get_current_curriculum_probs'):
            return self.smart_meter_data_loader.get_current_curriculum_probs(self.training_timestep)
        return {}

    # --------------------
    # Functions for saving log data / episode data / episode info etc.
    # --------------------

    def set_log_folder(self, log_folder: Path):
        """Set the log folder for this environment."""
        self.log_folder = log_folder
        
        # Create log folder if it doesn't exist
        if not self.log_folder.exists():
            self.log_folder.mkdir(parents=True)
        
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Log folder set to {self.log_folder}")

    def _save_episode_df(self, log_folder: Path, episode_idx: int):
        """
        Save the episode DataFrame to a pickle file.
        Args:
            log_folder (Path): The folder to save the episode DataFrame.
            episode_idx (int): The index of the episode to save.
        """
        target_folder = log_folder / "episodes_df"

        if not target_folder.exists():
            target_folder.mkdir(parents=True)

        episode_df_path = target_folder / f"episode_{episode_idx:0>4}.pkl"

        self.episode.df.to_pickle(episode_df_path)

        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode {episode_idx:0>4} DataFrame saved to {episode_df_path}")

    def _save_episode_info_list(self, log_folder: Path, episode_idx: int):
        """
        Save the infos of the whole episode to a json file with enhanced episode identification.
        Args:
            log_folder (Path): The folder to save the episode info.
            episode_idx (int): The index of the episode to save.
        """
        target_folder = log_folder / "episode_info"

        if not target_folder.exists():
            target_folder.mkdir(parents=True)

        # Get episode metadata for identification
        # episode_metadata = self.smart_meter_data_loader.get_episode_metadata(self.selected_idx)
        
        # Get episode content information from the episode DataFrame
        episode_content_id = None
        episode_length_days = None
        if hasattr(self.episode, 'df') and not self.episode.df.empty:
            if 'episode_content_id' in self.episode.df.columns:
                episode_content_id = self.episode.df['episode_content_id'].iloc[0]
            if 'episode_length_days' in self.episode.df.columns:
                episode_length_days = self.episode.df['episode_length_days'].iloc[0]
        
        # Enhanced episode metadata
        enhanced_metadata = {
            "episode_training_idx": int(episode_idx),  # Training sequence number
            "episode_data_idx": int(self.selected_idx),  # Index in data loader
            "episode_content_id": episode_content_id,  # Content-based unique identifier
            "episode_length_days": int(episode_length_days),  # Episode length in days
            "training_timestep": int(self.training_timestep),  # Current training timestep
            "dataset_type": str(self.mode.value),  # train/validate/test
            "curriculum_info": self.curriculum_info,  # Curriculum information
            "timestamp_created": datetime.now().isoformat()
        }
        
        # Save episode with enhanced metadata
        episode_info_with_metadata = {
            "metadata": enhanced_metadata,
            "episode_data": self.episode_info_list
        }

        episode_info_list_path = target_folder / f"episode_{episode_idx:0>4}_info.json"

        with open(episode_info_list_path, "w") as f:
            json.dump(episode_info_with_metadata, f, indent=4)

        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode {episode_idx:0>4} info saved to {episode_info_list_path}")
        if episode_content_id:
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode content ID: {episode_content_id}, Length: {episode_length_days} day(s)")

    def save_episodes_rewards(self, folder_path: Path):
        """
        Save the episodes rewards to a file.
        Args:
            folder_path (Path): The path to save the episodes rewards in JSON format.
        """
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        
        file_path = folder_path / "episodes_rewards.json"
        with open(file_path, "w") as f:
            json.dump(self.episodes_rewards, f, indent=4)
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episodes rewards saved to {file_path}")

    def save_environment_config(self, file_name: str = None):
        """Save environment configuration to a file."""
        if file_name is None:
            file_name = f"env_config_{self.mode.value}.json"
        
        env_config = {
            "mode": self.mode.value,
            "reward_lambda": self.reward_lambda,
            "battery_config": self.battery.get_battery_config(),
            "init_soc": float(self.init_soc),
            "h_network_type": str(self.h_network_rl_module.h_network_type),
            "data_loader_type": type(self.smart_meter_data_loader).__name__,
            "curriculum_enabled": self.curriculum_info.get('curriculum_enabled', False),
            "curriculum_info": self.curriculum_info,
            "render_mode": self.render_mode,
            "render_host": self.render_host,
            "render_port": self.render_port
        }
        
        env_config_path = self.log_folder / file_name
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
        
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Environment config saved to {env_config_path}")


    def close(self):
        """Cleanup - shared functionality."""
        # Save any remaining data
        if hasattr(self, 'episodes_rewards') and self.episodes_rewards:
            self.save_episodes_rewards()
    
        
        # Close render connection
        if self.render_connected and self.render_client_socket:
            try:
                # Send close message
                self._send_render_message(RenderWindowMessageType.CONTROL, {"command": RenderWindowControl.CLOSE.name})
                self.render_client_socket.close()
                self.render_connected = False
                self.render_client_socket = None
            except Exception as e:
                print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Error closing render client socket: {e}")

        
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Environment closed.")


class SmartMeterEnvFactory:
    """Factory class for creating Smart Meter environment instances."""
    
    @staticmethod
    def create(action_type: str, **kwargs) -> SmartMeterEnvironmentBase:
        """
        Create an environment instance based on action type.
        
        Args:
            action_type: Type of action space ('continuous' or 'discrete').
            **kwargs: Environment configuration parameters including:
                - smart_meter_data_loader: Data loader for smart meter data
                - h_network_rl_module: H-network RL module  
                - mode: Training mode (train/validate/test)
                - log_folder: Path for logging
                - rb_config: Battery configuration (optional)
                - reward_lambda: Reward weighting factor
                - render_mode: Render mode (optional)
                - render_host: Render server host
                - render_port: Render server port
                - aggregate_step_size: Step size for aggregate load quantization (discrete only)
                - battery_step_size: Step size for battery action quantization (discrete only)
                
        Returns:
            Environment instance appropriate for the action type.
            
        Raises:
            ValueError: If action_type is not supported.
        """
        if action_type.lower() == 'continuous':
            from rl_env.continuous.env_module import SmartMeterContinuousEnv
            # Filter out discrete-specific parameters for continuous environment
            continuous_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['aggregate_step_size', 'battery_step_size']}
            return SmartMeterContinuousEnv(**continuous_kwargs)
        elif action_type.lower() == 'discrete':
            from rl_env.discrete.env_module import SmartMeterDiscreteEnv
            return SmartMeterDiscreteEnv(**kwargs)
        else:
            raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")
