from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from collections import deque
import struct
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from render_window import RenderWindowControl, RenderWindowMessageType
from env_data_loader import SmartMeterDataLoader
from battery import RechargeableBattery
from model.H_network.h_network_arch import HNetworkType
from utils import print_log

from sklearn.preprocessing import StandardScaler

class SmartMeterEpisode:
    """
    Represents a single episode in the Smart Meter World environment.
    Contains the aggregate load data, battery state, and other relevant information.
    """
    def __init__(self, selected_aggregate_load_df: pd.DataFrame):
        self.df = selected_aggregate_load_df.copy()
        self.df['grid_load'] = None  # add a new column for grid load
        self.df['battery_soc'] = None  # add a new column for battery state of charge

        self.current_step = 0  # Current step in the episode

    def reset(self, selected_aggregate_load_df: pd.DataFrame, stdscalar: StandardScaler):
        """
        Reset the episode with a new aggregate load DataFrame.
        Args:
            selected_aggregate_load_df (pd.DataFrame): The DataFrame containing aggregate load data for the episode.
        """

        if stdscalar is None:
            raise ValueError("StandardScaler is not provided. Please provide a trained StandardScaler instance to standardize the aggregate load.")

        self.df = selected_aggregate_load_df.copy()
        self.df['grid_load'] = None
        self.df['battery_soc'] = None

        # standardize the aggregate load
        self.df['aggregate_std'] = stdscalar.transform(self.df[['aggregate']].values).flatten()

        self.current_step = 0

    def get_current_step(self) -> int:
        """
        Get the current step in the episode.
        Returns:
            int: The current step index.
        """
        return self.current_step
    
    def set_current_step(self, step: int):
        """
        Set the current step in the episode.
        Args:
            step (int): The step index to set.
        """
        if step < 0 or step >= len(self.df):
            raise ValueError("Step index out of bounds.")
        self.current_step = step
        
    def get_episode_length(self) -> int:
        """
        Get the length of the episode.
        Returns:
            int: The number of steps in the episode.
        """
        return len(self.df)
    
    def get_episode_info(self) -> dict:
        """
        Get information about the episode.
        Returns:
            dict: A dictionary containing episode information such as length and current step.
        """
        return {
            "length": self.get_episode_length(),
            "datetime_range": (self.df['datetime'].min(), self.df['datetime'].max()),
        }
   

class SmartMeterWorld(gym.Env):
    """
    Smart Meter World Environment for Hierarchical Reinforcement Learning.
    This environment simulates a energy management unit connected with a rechargeable battery.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, smart_meter_data_loader: SmartMeterDataLoader, h_model_type: HNetworkType, rb_config: Optional[dict] = None, render_mode=None, render_host='127.0.0.1', render_port=50007):
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
        self.episode = SmartMeterEpisode(self.smart_meter_data_loader.get_aggregate_load_segment(0))  # start with the first DataFrame

        # Initialize battery state
        self.battery = RechargeableBattery(**rb_config) if rb_config else RechargeableBattery(
                capacity=8.0,  # kWh
                max_charging_rate=4.0,  # kW
                max_discharging_rate=4.0,  # kW
                efficiency=1.0,  # 100% efficiency
                init_soc=0.5  # 50% initial state of charge
        )

        # define state space and action space
        self.observation_space = gym.spaces.Dict({
            # "aggregate_load": gym.spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),    # the understandardized aggregate load in W
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

        self.reward_lambda = 0.5        # value between [0,1]. Closer to zero -> more privacy focused

        self.h_network_type = h_model_type
        self.h_network = None  # Placeholder for H-network, to be loaded or trained
        self.h_network_stdscaler = None  # Placeholder for H-network standard scaler, to be loaded or trained
        self.h_network_criterion = torch.nn.GaussianNLLLoss(reduction='none')  # loss function as the privacy signal. Using sum mode
        self.h_network_inference_buffer = deque()  # Buffer for H-network inference, to store recent pair of input (z_t, y_t) and desired target (y_{t+1})
        self.H_NETWORK_MAXSEQLEN = 512  # Maximum sequence length for H-network inference buffer


        # render stuffs
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            # plt.ion()  # Enable interactive mode for matplotlib
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to the initial state.
        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for resetting the environment.
        Returns:
            dict: Initial observation of the environment.
        """
        super().reset(seed=seed, options=options)

        self.battery.reset(self.np_random.uniform(0.0, 1.0))    # randomize the battery state of charge

        # randomly select an aggregate load DataFrame from the list
        selected_idx = self.np_random.integers(0, self.smart_meter_data_loader.get_divided_segments_length())
        self.episode.reset(self.smart_meter_data_loader.get_aggregate_load_segment(selected_idx), self.h_network_stdscaler)  # Reset with a new episode

        print_log(f"[SmartMeterWorld] Resetting environment with a new episode. Episode info: {self.episode.get_episode_info()}")

        self.h_network_inference_buffer.clear()  # Clear the inference buffer for H-network

        observation = self._get_obs()
        info = self._get_info(observation)
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
        # TODO: proofread later

        g_signal = self._g_signal(
            s_t_datetime=self.episode.df.iloc[current_step]['datetime'],
            s_t_plus_1_datetime=self.episode.df.iloc[current_step + 1]['datetime'],
            power_kw=power_charged_discharged
        )

        f_signal = self._f_signal(
            y_t=y_t,
            z_t=z_t,
            y_t_plus_1=self.episode.df.iloc[current_step + 1]['aggregate']
        )

        # reward function is the negative of loss signal
        reward = - (self.reward_lambda * g_signal + (1 - self.reward_lambda) * f_signal)

        # determine termination condition
        terminated = current_step >= self.episode.get_episode_length() - 1 - 1      # -1 because we want to stop before the last step to avoid index out of range

        if not terminated:
            # Update the current step
            self.episode.set_current_step(current_step + 1)

        next_obs = self._get_obs()
        return next_obs, \
            reward, \
            terminated, \
            False, \
            self._get_info(
                obs=next_obs, power_kw=power_kw, power_charged_discharged=power_charged_discharged, reward=reward, f_signal=f_signal, g_signal=g_signal
            )

    def _get_info(self, obs, power_kw=None, power_charged_discharged=None, reward=None, f_signal=None, g_signal=None):
        """
        Acquire additional information about the current state of the environment for debugging or logging purposes.

        Returns:
            dict: A dictionary containing additional information about the current state of the environment.
        """

        return {
            "current_step": self.episode.get_current_step(),
            "battery_soc (kWh)": obs["battery_soc"][0],
            "user_load (W)": obs["aggregate_load"][0],
            "(prev) grid_load (W)": self.episode.df.iloc[self.episode.get_current_step() - 1]['grid_load'] if self.episode.get_current_step() > 0 else None,
            "last_action (kW)": power_kw if power_kw is not None else None,
            "last_battery_actiuon (kW)": power_charged_discharged if power_charged_discharged is not None else None,
            "last_reward" : reward if reward is not None else None,
            "last_f_signal": f_signal if f_signal is not None else None,
            "last_g_signal": g_signal if g_signal is not None else None,
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
        """
        
        if self.h_network is None:
            raise ValueError("H-network is not set. Please set the H-network before calling this method.")
        
        if self.h_network_stdscaler is None:
            raise ValueError("H-network standard scaler is not set. Please set the H-network standard scaler before calling this method.")

        # Store the current observation in the buffer for H-network inference
        self.h_network_inference_buffer.append(((z_t, y_t), y_t_plus_1))        # ((grid load, user load), next user load)

        h_network_input = np.array([item[0] for item in self.h_network_inference_buffer], dtype=np.float32)
        h_network_target = np.array([item[1] for item in self.h_network_inference_buffer], dtype=np.float32)

        # scale the inputs and targets
        for i in range(h_network_input.shape[1]):
            if h_network_input[:, i].shape[0] == 1:  # if there is single sample
                h_network_input[:, i] = self.h_network_stdscaler.transform(h_network_input[:, i].reshape(1, -1)).flatten()
            else:
                # the reshape is to scale each feature independently
                h_network_input[:, i] = self.h_network_stdscaler.transform(h_network_input[:, i].reshape(-1, 1)).flatten()

        if h_network_target.shape[0] == 1:  # if there is single sample
            h_network_target = self.h_network_stdscaler.transform(h_network_target.reshape(1, 1)).flatten()
        else:
            h_network_target = self.h_network_stdscaler.transform(h_network_target.reshape(-1, 1)).flatten()


        # Chunk, pad, and mask the input and target sequences, 
        # break the input and target into chunks if they exceed the maximum sequence length
        if len(h_network_input) > self.H_NETWORK_MAXSEQLEN:
            h_network_input, h_network_target, h_network_mask = self._chunk_pad_mask_sequences(
                h_network_input.reshape(1, -1, 2),  # reshape to (N, T, 2) where N=1 for single sequence
                h_network_target.reshape(1, -1),  # reshape to (N, T)
                chunk_size=self.H_NETWORK_MAXSEQLEN,
                stride=self.H_NETWORK_MAXSEQLEN,
            )
        else:
            # no padding is needed
            h_network_input, h_network_target, h_network_mask = h_network_input.reshape(1, -1, 2), h_network_target.reshape(1, -1), np.ones(len(h_network_input), dtype=bool).reshape(1, -1)
            
            # Convert to torch tensors
            h_network_input = torch.tensor(h_network_input, dtype=torch.float32)
            h_network_target = torch.tensor(h_network_target, dtype=torch.float32)
            h_network_mask = torch.tensor(h_network_mask, dtype=torch.bool)

        # temporary dataloader for H-network inference
        dataloader = DataLoader(
            TensorDataset(h_network_input, h_network_target, h_network_mask),
            batch_size=64,  # Adjust batch size as needed
            shuffle=False,  # No need to shuffle for inference
        )

        signal_output = None

        # Get the H-network reward
        with torch.no_grad():

            lstm_1_h, lstm_2_h = None, None

            for i, batch in enumerate(dataloader):
                _input, _target, _mask = batch

                if self.h_network_type == HNetworkType.H_NETWORK:
                    # H-Network that predicts only the mean
                    mean, lstm_1_h, lstm_2_h = self.h_network(_input, lstm_1_h, lstm_2_h)
                elif self.h_network_type == HNetworkType.H_NETWORK2:
                    # H-Network that predicts both mean and log variance
                    mean, log_var, lstm_1_h, lstm_2_h = self.h_network(_input, lstm_1_h, lstm_2_h)

                if i != len(dataloader) - 1:
                    # If not the last batch, we do not compute the loss yet
                    continue

                # Apply the mask to the mean and target
                mean = mean.view(-1)  # flatten the mean
                _target = _target.view(-1)  # flatten the target
                _mask = _mask.view(-1)  # flatten the mask

                mean_masked = torch.masked_select(mean, _mask)  # apply the mask to the mean
                _target_masked = torch.masked_select(_target, _mask)  # apply the mask to the target

                if self.h_network_type == HNetworkType.H_NETWORK:
                    loss = self.h_network_criterion(mean_masked, _target_masked, var=torch.ones_like(mean_masked))  # assuming unit variance

                elif self.h_network_type == HNetworkType.H_NETWORK2:
                    # For H-Network2, we need to compute the variance from log variance
                    log_var_masked = torch.masked_select(log_var.view(-1), _mask)                                     # apply the mask to the log variance
                    loss = self.h_network_criterion(mean_masked, _target_masked, var=torch.exp(log_var_masked))    # compute the loss with the masked mean and target

                # compute likelihood of each sample
                likelihood = torch.exp(-loss)

                # signal output is expected value of the negative loss
                signal_output = torch.mul(likelihood, -loss)

        # return the loss. This approximates \mathbb{E} \log(p(y_t_plus_1 | z_t, y_t)) -> the sum of signals approximates the Mutual Information (MI)
        signal_output = signal_output[-1]        # only consider the last sample.
        return signal_output.item()
    
    def _chunk_pad_mask_sequences(self, input_sequences, target_sequences, chunk_size=512, stride=64, padding_value=0.0):
        """
        Chunk, pad, and mask input/target sequences for LSTM training.

        Copied from h_network_training.ipynb
        Args:
            input_sequences: list of np.ndarray, each shape (T, 2)
            target_sequences: list of np.ndarray, each shape (T,)
            chunk_size: int, length of each chunk
            stride: int, step size for rolling window
            padding_value: value to use for padding
        Returns:
            padded_inputs: torch.Tensor, shape (N, chunk_size, 2)
            padded_targets: torch.Tensor, shape (N, chunk_size)
            mask: torch.BoolTensor, shape (N, chunk_size)
        """
        chunked_inputs = []
        chunked_targets = []
        chunked_masks = []

        for inp_seq, tgt_seq in zip(input_sequences, target_sequences):
            seq_len = inp_seq.shape[0]
            # Rolling window chunking
            for start in range(0, seq_len, stride):
                end = start + chunk_size
                inp_chunk = inp_seq[start:end]
                tgt_chunk = tgt_seq[start:end]
                mask = np.zeros(chunk_size, dtype=bool)
                valid_len = min(chunk_size, seq_len - start)
                mask[:valid_len] = True
                # Pad if needed
                if inp_chunk.shape[0] < chunk_size:
                    pad_len = chunk_size - inp_chunk.shape[0]
                    inp_chunk = np.pad(inp_chunk, ((0, pad_len), (0, 0)), mode='constant', constant_values=padding_value)
                    tgt_chunk = np.pad(tgt_chunk, (0, pad_len), mode='constant', constant_values=padding_value)
                chunked_inputs.append(torch.tensor(inp_chunk, dtype=torch.float32))
                chunked_targets.append(torch.tensor(tgt_chunk, dtype=torch.float32))
                chunked_masks.append(torch.tensor(mask, dtype=torch.bool))
                # Stop if we've reached the end
                if end >= seq_len:
                    break

        padded_inputs = torch.stack(chunked_inputs)
        padded_targets = torch.stack(chunked_targets)
        mask = torch.stack(chunked_masks)
        return padded_inputs, padded_targets, mask




    def set_h_network(self, h_network):
        """
        Set the H-network for the environment.
        Args:
            h_network: The H-network model to be used in the environment.
        """
        self.h_network = h_network

    def set_h_network_stdscaler(self, h_network_stdscaler):
        """
        Set the H-network standard scaler for the environment.
        Args:
            h_network_stdscaler: The standard scaler to be used for H-network inputs.
        """
        self.h_network_stdscaler = h_network_stdscaler

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

    def save_graph(self, path:str):
        """Request the render window to save the current graph to the given path."""
        self._send_control_message(RenderWindowControl.SAVE_GRAPH.name, {'path': path})

    def reset_render_window(self):
        """Request the render window to reset (clear) all buffers."""
        self._send_control_message(RenderWindowControl.RESET.name)

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