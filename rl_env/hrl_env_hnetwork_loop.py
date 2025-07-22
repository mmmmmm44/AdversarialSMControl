import numpy as np
import gymnasium as gym
import pandas as pd

from datetime import datetime, timedelta, timezone, time
from typing import Optional
from collections import deque
import struct
import json
from pathlib import Path
import math

from render_window import RenderWindowControl, RenderWindowMessageType
from env_data_loader import SmartMeterDataLoader
from battery import RechargeableBattery
from model.H_network.h_network_rl_module import HNetworkRLModule
from hrl_env_episode import SmartMeterEpisode
from utils import print_log
   

class SmartMeterWorld(gym.Env):
    """
    Smart Meter World Environment for Hierarchical Reinforcement Learning.
    This environment simulates a energy management unit connected with a rechargeable battery.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, smart_meter_data_loader: SmartMeterDataLoader, h_network_rl_module: HNetworkRLModule, log_folder: Path, rb_config: Optional[dict] = None, reward_lambda: float = 0.5, render_mode=None, render_host='127.0.0.1', render_port=50007):
        '''
        Initializes the SmartMeterWorld environment.
        
        Args:
            smart_meter_data_loader (SmartMeterDataLoader): Data loader for smart meter data.
            h_network_rl_module (HNetworkRLModule): H-network RL module for providing per-step privacy-related signals.
            log_folder (Path): Folder to save logs and results.
            rb_config (Optional[dict]): Configuration for the rechargeable battery. If None, default values are used.
            reward_lambda (float): Weighting factor for the reward function, between 0 and 1.
            render_mode (Optional[str]): Render mode for visualization. If 'human', it connects to a render server.
            render_host (str): Host address of the render server.
            render_port (int): Port number of the render server.
        '''
        super(SmartMeterWorld, self).__init__()

        # TCP client for real-time rendering
        self.render_host = render_host
        self.render_port = render_port
        self.render_client_socket = None
        self.render_connected = False

        self.smart_meter_data_loader = smart_meter_data_loader
        if not self.smart_meter_data_loader:
            raise ValueError("No SmartMeterDataLoader provided.")

        # initialize an episode
        self.selected_idx = 0
        self.episode = SmartMeterEpisode(self.smart_meter_data_loader.get_aggregate_load_segment(self.selected_idx))  # start with the first DataFrame
        self.episode_info_list = []

        # Initialize battery state
        # init_soc is the constant initial state of charge (SoC) of the battery for each episode beginning
        self.init_soc = 0.15 if rb_config is None else rb_config.get('init_soc', 0.5)  # default to 50% if not specified
        self.battery = RechargeableBattery(**rb_config) if rb_config else RechargeableBattery(
                capacity=8.0,  # kWh
                max_charging_rate=4.0,  # kW
                max_discharging_rate=4.0,  # kW
                efficiency=1.0,  # 100% efficiency
                init_soc=self.init_soc  # 50% initial state of charge
        )

        # define state space and action space
        self.observation_space = gym.spaces.Dict({
            "aggregate_load": gym.spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32),    # standardized aggregate load
            "battery_soc": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "timestamp_features": gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)  # Hour, Day of Week, Month
        })
    
        # high-level policy action space: decide whether right now is a good time to mimic a particular appliance's load pattern
        # and if mimick, which appliance to mimic
        # self.high_level_action_space = gym.spaces.Discrete(2N+1)        # N is number of appliances in the load database

        # low-level policy action space: charging or discharging power in kW
        # normalized to [-1, 1] where -1 is max discharging, 0 is no action, and 1 is max charging
        self.low_level_action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.action_space = self.low_level_action_space

        self.reward_lambda = reward_lambda        # value between [0,1]. Closer to zero -> more privacy focused

        # H-network stuffs
        self.h_network_rl_module = h_network_rl_module
        if not self.h_network_rl_module:
            raise ValueError("No HNetworkRLModule provided.")    
        self.h_network_episode_inference_buffer = deque()

        # reward logging stuffs
        self.log_folder = log_folder
        self.episodes_rewards = []  # to store the sum of rewards, mean of rewards, and std. of rewards
        self.per_episode_rewards = []  # to store the rewards of the current episode

        # render stuffs
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            # Try to connect to the render server
            self._init_render_client()

            print_log(f"[SmartMeterWorld] Render mode set to '{self.render_mode}'. Render server at {self.render_host}:{self.render_port}. render_connected: {self.render_connected}. render_client_socket: {self.render_client_socket}")

    def _init_render_client(self):
        import socket
        try:
            self.render_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.render_client_socket.connect((self.render_host, self.render_port))
            self.render_connected = True
        except Exception as e:
            print_log(f"[SmartMeterWorld] Could not connect to render server: {e}")
            self.render_connected = False

    def _get_obs(self):
        """
        Get the current observation of the environment.
        Returns:
            dict: Current state observation including aggregate load, battery state of charge, and timestamp features.
        """
        current_step = self.episode.get_current_step()

        soc = self.battery.get_normalized_state_of_charge()
        current_load = self.episode.df.iloc[current_step]['aggregate_std']      # receive standardized aggregate load may be better?
        timestamp = self.episode.df.iloc[current_step]['timestamp']
        timestamp_features = self._create_timestamp_features(timestamp)

        return {
            "aggregate_load": np.array([current_load], dtype=np.float32),
            "battery_soc": np.array([soc], dtype=np.float32),
            "timestamp_features": timestamp_features
        }
    
    def reset(self, episode_idx: int = None, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to the initial state.
        Args:
            episode_idx (int): The index of the episode to reset (begin) with.
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for resetting the environment.
        Returns:
            dict: Initial observation of the environment.
        """
        super().reset(seed=seed, options=options)

        self.h_network_episode_inference_buffer.clear()

        self.battery.reset(self.init_soc)    # Reset the battery state of charge to initial value

        # reset the episodes rewards
        self.per_episode_rewards = []

        # select an episode to reset with
        # if episode_idx is None, randomly select an episode from the available episodes
        # if episode_idx is provided, check if it is valid and use it; otherwise,
        # randomly select an aggregate load DataFrame from the list
        if episode_idx is not None and (episode_idx < 0 or episode_idx >= self.smart_meter_data_loader.get_divided_segments_length()):
            raise ValueError(f"Invalid episode index: {episode_idx}. Must be between 0 and {self.smart_meter_data_loader.get_divided_segments_length() - 1}.")

        self.selected_idx = episode_idx if episode_idx is not None else int(self.np_random.integers(0, self.smart_meter_data_loader.get_divided_segments_length()))
        self.episode = SmartMeterEpisode(self.smart_meter_data_loader.get_aggregate_load_segment(self.selected_idx))  # Reset with a new episode
        self.episode_info_list = []

        print_log(f"[SmartMeterWorld] Resetting environment with a new episode. Episode info: {self.episode.get_episode_info()}")

        self.send_env_info()  # send the environment info to the render window

        observation = self._get_obs()
        info = self._get_info(observation)

        self.episode_info_list.append(info)         # store the initial observation info

        return observation, info
    

    def step(self, action):
        """
        Execute one time step within the environment.
        Args:
            action: Action taken by the agent (charging or discharging power).
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        current_step = self.episode.get_current_step()

        # get user load
        y_t = self.episode.df.iloc[current_step]['aggregate']

        # Get the action from the agent
        power_kw_normalized = action[0]
        power_kw = self.battery.compute_unnormalized_charge(power_kw_normalized)  # convert normalized action to kW

        # compute the duration of the step in seconds
        if current_step < self.episode.get_episode_length() - 1:
            duration = self.episode.df.iloc[current_step + 1]['timestamp'] - self.episode.df.iloc[current_step]['timestamp']      # timestamp is in seconds as unit
        else:
            duration = 6
        
        # before applying the action, update the battery state of charge in the dataframe
        self.episode.df.iat[current_step, self.episode.df.columns.get_loc('battery_soc')] = self.battery.get_normalized_state_of_charge()

        # Apply the action to the battery
        power_charged_discharged = self.battery.charge_discharge(power_kw, duration, y_t)

        # --------------------
        # after applying the action
        # advance to the next step, but we have to calculate the grid load
        # hence y_{t} is the aggregated user load, and we compare whether our battery action can help mask the user load y_{t+1}, given previous (masked) grid load z_{t} (an exposed information that may be acquired by attacker)
        # --------------------

        # compute tentative grid load
        z_t = y_t  + power_charged_discharged * 1000   # convert kW to W
        
        z_t = np.clip(z_t, 0, None)  # ensure grid load is non-negative
        self.episode.df.iat[current_step, self.episode.df.columns.get_loc('grid_load')] = z_t  # update the grid load in the dataframe

        # reward function
        # logic: when computing the reward, we are already passed `duration` number of seconds, so we can use the next step's aggregate load to compute the reward. (as the next step's aggregate load will be our observation)

        g_signal = self._g_signal(
            s_t_datetime=self.episode.df.iloc[current_step]['datetime'],
            s_t_plus_1_datetime=self.episode.df.iloc[current_step + 1]['datetime'],
            power_kw=power_charged_discharged
        )

        f_signal, f_signal_additional_info = self._f_signal(
            y_t=y_t,
            z_t=z_t,
            y_t_plus_1=self.episode.df.iloc[current_step + 1]['aggregate']
        )

        # reward function is the negative of loss signal
        reward = - (self.reward_lambda * g_signal + (1 - self.reward_lambda) * f_signal)
        self.per_episode_rewards.append(reward)  # logging use; append the reward to the current episode rewards

        # determine termination condition
        terminated = current_step >= self.episode.get_episode_length() - 1 - 1      # -1 because we want to stop before the last step to avoid index out of range

        if not terminated:
            # Update the current step
            self.episode.set_current_step(current_step + 1)
        
        next_obs = self._get_obs()
        next_info = self._get_info(
            obs=next_obs, power_kw=power_kw, power_charged_discharged=power_charged_discharged, reward=reward, f_signal=f_signal, g_signal=g_signal, f_signal_additional_info=f_signal_additional_info
        )
        self.episode_info_list.append(next_info)

        # log the episode when terminated
        if terminated:
            # push the old episode to the HNetworkRLModule buffer
            self.h_network_rl_module.push_to_replay_buffer(self.episode)

            # calculate the sum of rewards for the episode
            episode_sum = math.fsum(self.per_episode_rewards)
            episode_reward_stats = {
                "sum": float(episode_sum),
                "mean": float(np.mean(self.per_episode_rewards)),
                "std": float(np.std(self.per_episode_rewards))
            }
            self.episodes_rewards.append(episode_reward_stats)
            self.per_episode_rewards = []  # reset the per-episode rewards for the next episode

            self._save_episode_info_list(self.log_folder, len(self.episodes_rewards))

            print_log(f"[SmartMeterWorld] Episode finished. Sum of rewards: {episode_sum}. Mean of rewards: {episode_reward_stats['mean']}. Std of rewards: {episode_reward_stats['std']}")

        return next_obs, \
            reward, \
            terminated, \
            False, \
            next_info

    def _get_info(self, obs, power_kw=None, power_charged_discharged=None, reward=None, f_signal=None, g_signal=None, f_signal_additional_info=None):
        """
        Acquire additional information about the current state of the environment for debugging or logging purposes.

        Returns:
            dict: A dictionary containing additional information about the current state of the environment.
        """

        # type casting is necessary to ensure the values are JSON serializable

        info_dict = {
            "episode_index": int(self.selected_idx),
            "current_step": int(self.episode.get_current_step()),
            'datetime': self.episode.df.iloc[self.episode.get_current_step()]['datetime'].isoformat(timespec='seconds') if self.episode.get_current_step() < len(self.episode.df) else None,
            "battery_soc (%)": float(obs["battery_soc"][0]),
            "battery_soc (kWh)": float(obs["battery_soc"][0]) * self.battery.capacity,  # convert normalized SoC to kWh
            "user_load (W)": float(self.episode.df.iloc[self.episode.get_current_step()]['aggregate']) if self.episode.get_current_step() >= 0 else None,
            "grid_load (W)": float(self.episode.df.iloc[self.episode.get_current_step() - 1]['grid_load']) if self.episode.get_current_step() > 0 else None,
            "action (kW)": float(power_kw) if power_kw is not None else None,
            "battery_action (kW)": float(power_charged_discharged) if power_charged_discharged is not None else None,
            "reward" : float(reward) if reward is not None else None,
            "f_signal": float(f_signal) if f_signal is not None else None,
            "g_signal": float(g_signal) if g_signal is not None else None,
        }
        info_dict.update({"f_signal" + "-" + k : float(v) for k, v in f_signal_additional_info.items()} if f_signal_additional_info is not None else {})

        return info_dict
    
    def get_env_info(self):
        """
        Get information about the general setting of the environment.
        Returns:
            dict: A dictionary containing environment information such as selected episode index and current step.
        """
        return {
            "reward_lambda": self.reward_lambda,
        }

    def _create_timestamp_features(self, timestamp: int) -> np.ndarray:
        """
        Create timestamp features for the environment. This is inspired by DiffNILM paper.
        Args:
            timestamp (int): Timestamp in seconds since epoch.
        Returns:
            np.ndarray: Normalized features representing hour of day, day of week, and month.
        """
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return np.array([
            dt.hour / 24.0 - 0.5,  # Hour of day normalized
            dt.weekday() / 6.0 - 0.5,  # Day of week normalized, where Monday is 0 and Sunday is 6
            dt.month / 12.0 - 0.5  # Month normalized
        ], dtype=np.float32)

    def _g_signal(self, s_t_datetime:datetime, s_t_plus_1_datetime:datetime, power_kw: float) -> float:
        """
        Calculate the cost signal based on elapsed time and power used.
        Args:
            s_t_datetime (datetime): Current timestamp.
            s_t_plus_1_datetime (datetime): Next timestamp.
            power_kw (float): Power used in kW.
        Returns:
            float: Calculated cost signal.
        """

        # return cost incurred for the power used in the time period

        return self._get_weighted_electricity_cost(s_t_datetime, s_t_plus_1_datetime) * abs(power_kw)
    
    def _get_weighted_electricity_cost(self, s_t_datetime: datetime, s_t_plus_1_datetime: datetime) -> float:
        """
        Calculate the weighted electricity cost based on a time-of-use pricing model.
        Args:
            s_t_datetime (datetime): Current timestamp.
            s_t_plus_1_datetime (datetime): Next timestamp.
        Returns:
            float: Weighted electricity cost, across the time period. (i.e. delta_t * avg_price_per_kWh)
        """
        # we implement a time-of-use pricing model
        # declare the time-of-use pricing model
        time_of_use_prices = {
            (time(0,0,0,0), time(7,0,0,0)): 0.101,  # off-peak price
            (time(7,0,0,0), time(11,0,0,0)): 0.208,   # peak price
            (time(11,0,0,0), time(17,0,0,0)): 0.144,  # mid-peak price
            (time(17,0,0,0), time(19,0,0,0)): 0.208,   # peak price
            (time(19,0,0,0), time(23,59,59,999999)): 0.101,  # evening price
        }

        # compute the weighted electricity cost
        total_cost = 0.0
        for start_time, end_time in time_of_use_prices.keys():
            if s_t_datetime.time() <= end_time and s_t_plus_1_datetime.time() >= start_time:
                # Calculate the overlap between the time intervals
                overlap_start = max(s_t_datetime, datetime.combine(s_t_datetime.date(), start_time))
                overlap_end = min(s_t_plus_1_datetime, datetime.combine(s_t_plus_1_datetime.date(), end_time))
                
                if overlap_start < overlap_end:
                    overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600.0  # Convert to hours
                    price_per_kwh = time_of_use_prices[(start_time, end_time)]
                    total_cost += overlap_duration * price_per_kwh

        return total_cost

    
    def _f_signal(self, y_t: float, z_t: float, y_t_plus_1: float) -> float:
        """
        Calculate the H-network reward based on the user's load and load received from the grid
        Args:
            y_t (float): User's load at time t.
            z_t (float): Load received from the grid at time t.
            y_t_plus_1 (float): User's load at time t+1.
        Returns:
            float: Calculated H-network reward.
            dict: Additional information about the last predictions.
        """
        
        if self.h_network_rl_module is None:
            raise ValueError("H-network module is not set. Please set the H-network module before calling this method.")
        
        # create h_network_input and h_network_target sequence
        self.h_network_episode_inference_buffer.append(((y_t, z_t), y_t_plus_1))
        h_network_input = np.array([item[0] for item in self.h_network_episode_inference_buffer], dtype=np.float32)
        h_network_target = np.array([item[1] for item in self.h_network_episode_inference_buffer], dtype=np.float32)

        # standardize both h_network_input and h_network_target using the standard scaler in the episode object
        # scale the inputs and targets
        standard_scalar = self.episode.standard_scalar
        for i in range(h_network_input.shape[1]):
            if h_network_input[:, i].shape[0] == 1:  # if there is single sample
                h_network_input[:, i] = standard_scalar.transform(h_network_input[:, i].reshape(1, -1)).flatten()
            else:
                # the reshape is to scale each feature independently
                h_network_input[:, i] = standard_scalar.transform(h_network_input[:, i].reshape(-1, 1)).flatten()

        if h_network_target.shape[0] == 1:  # if there is single sample
            h_network_target = standard_scalar.transform(h_network_target.reshape(1, 1)).flatten()
        else:
            h_network_target = standard_scalar.transform(h_network_target.reshape(-1, 1)).flatten()


        # call HNetworkRLModule.compute_f_signal to compute the f_signal and return it.
        f_signal, additional_info = self.h_network_rl_module.compute_f_signal(h_network_input, h_network_target)
        return f_signal, additional_info
    
    def _save_episode_info_list(self, log_folder:Path, episode_idx:int):
        """
        Save the infos of the whole episode to a json file
        Args:
            log_folder (Path): The folder to save the episode info.
            episode_idx (int): The index of the episode to save.
        """
        target_folder = log_folder / "episode_info"

        if not target_folder.exists():
            target_folder.mkdir(parents=True)

        episode_info_list_path = target_folder / f"episode_{episode_idx:0>4}_info.json"

        with open(episode_info_list_path, "w") as f:
            json.dump(self.episode_info_list, f, indent=4)

        print_log(f"Episode {episode_idx:0>4} info saved to {episode_info_list_path}")

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
        print_log(f"[SmartMeterWorld] Episodes rewards saved to {file_path}")

    def _create_env_config(self):
        """
        Create the environment configuration dictionary.
        Returns:
            dict: Environment configuration including battery settings, reward lambda, and H-network module settings.
        """
        env_config = {
            "battery": self.battery.get_battery_config(),
            "reward_lambda": self.reward_lambda,
            "h_network_type": str(self.h_network_rl_module.h_network_type),
            "init_soc": self.init_soc,
        }
        return env_config

    def save_env_config(self, log_folder: Path):
        """
        Save the environment configuration to a file.
        Args:
            experiment_folder (Path): The folder to save the environment config.
        """
        if not log_folder.exists():
            log_folder.mkdir(parents=True)

        env_config_path = log_folder / "env_config.json"
        with open(env_config_path, "w") as f:
            json.dump(self._create_env_config(), f, indent=4)

        print_log(f"[SmartMeterWorld] Environment config saved to {env_config_path}")

    def render(self):
        """
        Render the environment as a line plot, showing the aggregate load and battery state of charge.
        If render_mode is 'human', send the figure to the render server via TCP.
        """
        if self.render_mode == "human":
            self._plot_graphs(send_tcp=True)

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
                print_log(f"[SmartMeterWorld] Failed to send message to render server: {e}")

    def _plot_graphs(self, send_tcp=False):
        """
        Instead of plotting, send the latest data as a JSON dict to the render window if send_tcp is True.
        """
        if send_tcp and self.render_connected and self.render_client_socket:

            current_step = self.episode.get_current_step()

            dt = self.episode.df['datetime'].iat[current_step - 1]
            user_load = self.episode.df['aggregate'].iat[current_step - 1]
            grid_load = self.episode.df['grid_load'].iat[current_step - 1]
            battery_soc = self.episode.df['battery_soc'].iat[current_step - 1]
            message = {
                'type': RenderWindowMessageType.DATA.name.lower(),
                'payload': {
                    'timestamp': dt.isoformat() if not pd.isnull(dt) else None,
                    'user_load': float(user_load) if user_load is not None else None,
                    'grid_load': float(grid_load) if grid_load is not None else None,
                    'battery_soc': float(battery_soc) if battery_soc is not None else None
                }
            }
            self._send_json_message(message)

    def _send_control_message(self, command, payload=None):
        """Send a control message to the render window."""
        message = {
            'type': RenderWindowMessageType.CONTROL.name.lower(),
            'command': command,
        }
        if payload is not None:
            message['payload'] = payload
        self._send_json_message(message)

    def save_graph(self, kwargs):
        """Request the render window to save the current graph to the given path.
        
        Args:
            kwargs: Keyword arguments to pass to the render window's save_graph method, which are valid arguments for plt.savefig().
        """
        self._send_control_message(RenderWindowControl.SAVE_GRAPH.name, kwargs)

    def reset_render_window(self):
        """Request the render window to reset (clear) all buffers."""
        self._send_control_message(RenderWindowControl.RESET.name)

    def send_env_info(self):
        """Send environment information to the render window."""
        env_info = {
            'selected_idx': self.selected_idx,
        }
        self._send_control_message(RenderWindowControl.RECEIVE_ENV_INFO.name, {'env_info': env_info})

    def close(self):
        """
        Close the environment and the render client socket if it exists. Also notify the render window.
        """
        if self.render_client_socket:
            self._send_control_message(RenderWindowControl.CLOSE.name)

            try:
                self.render_client_socket.close()
            except Exception as e:
                print_log(f"[SmartMeterWorld] Error closing render client socket: {e}")
            self.render_client_socket = None
            self.render_connected = False

        print_log("[SmartMeterWorld] Environment closed.")

    def terminate_render_window(self):
        """Request the render window to terminate itself. This will also close the render client socket."""
        self._send_control_message(RenderWindowControl.CLOSE.name)
        self._send_control_message(RenderWindowControl.TERMINATE.name)