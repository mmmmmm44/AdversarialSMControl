from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta, timezone, time
from collections import deque

import torch


from battery import RechargeableBattery

class SmartMeterWorld(gym.Env):
    """
    Smart Meter World Environment for Hierarchical Reinforcement Learning.
    This environment simulates a energy management unit connected with a rechargeable battery.
    """

    def __init__(self, aggregate_load: pd.DataFrame, rb_config: Optional[dict] = None):
        super(SmartMeterWorld, self).__init__()

        self.aggregate_load = aggregate_load

        # Initialize battery state
        self.battery = RechargeableBattery(**rb_config) if rb_config else RechargeableBattery(
                capacity=8.0,  # kWh
                max_charging_rate=4.0,  # kW
                max_discharging_rate=4.0,  # kW
                efficiency=0.99,  # 99% efficiency
                init_soc=0.5  # 50% initial state of charge
        )

        self.current_step = 0

        # define state space and action space
        self.observation_space = gym.spaces.Dict({
            "aggregate_load": gym.spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),    # TODO: look back to the range of the aggregated load
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
            dtype=np.float32        # int or float?
        )

        self.action_space = self.low_level_action_space

        self.reward_lambda = 0.5  # lambda for the reward function, can be tuned

        self.h_network = None  # Placeholder for H-network, to be loaded or trained
        self.h_network_stdscaler = None  # Placeholder for H-network standard scaler, to be loaded or trained
        self.h_network_criterion = torch.nn.GaussianNLLLoss()  # loss function as the privacy signal
        self.h_network_inference_buffer = deque(maxlen=512)  # Buffer for H-network inference, to store recent pair of input (z_t, y_t) and desired target (y_{t+1})

    def _get_obs(self):
        """
        Get the current observation of the environment.
        Returns:
            dict: Current state observation including aggregate load, battery state of charge, and timestamp features.
        """
        soc = self.battery.get_state_of_charge()
        current_load = self.aggregate_load.iloc[self.current_step]['aggregate']
        timestamp = self.aggregate_load.iloc[self.current_step]['timestamp']
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
        self.current_step = 0
        self.battery.reset(self.np_random.uniform(0.0, 1.0))    # randomize the battery state of charge

        self.h_network_inference_buffer.clear()  # Clear the inference buffer for H-network

        observation = self._get_obs()
        return observation, {}      # no info dictionary needed for now
    

    def step(self, action):
        """
        Execute one time step within the environment.
        Args:
            action: Action taken by the agent (charging or discharging power).
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Apply the action to the battery
        power_kw_normalized = action[0]
        power_kw = self.battery.compute_unnormalized_charge(power_kw_normalized)  # convert normalized action to kW

        # compute the duration of the step in seconds
        if self.current_step < len(self.aggregate_load) - 1:
            duration = self.aggregate_load.iloc[self.current_step + 1]['timestamp'] - self.aggregate_load.iloc[self.current_step]['timestamp']      # timestamp is in seconds as unit
        else:
            duration = 6
        
        # Apply the action to the battery
        self.battery.charge(power_kw, duration=duration)

        # compute grid load
        z_t = self.aggregate_load.iloc[self.current_step]['aggregate']  + power_kw * 1000   # convert kW to W
        z_t = np.clip(z_t, 0, None)  # ensure grid load is non-negative

        # reward function
        # TODO: proofread later

        g_reward = self._g_reward(
            s_t_datetime=self.aggregate_load.iloc[self.current_step]['datetime'],
            s_t_plus_1_datetime=self.aggregate_load.iloc[self.current_step + 1]['datetime'],
            power_kw=power_kw
        )

        f_reward = self._f_reward(
            y_t=self.aggregate_load.iloc[self.current_step]['aggregate'],
            z_t=z_t,
            y_t_plus_1=self.aggregate_load.iloc[self.current_step + 1]['aggregate']
        )

        reward = (1 - self.reward_lambda) * g_reward + self.reward_lambda * f_reward

        # determine termination condition
        terminated = self.current_step >= len(self.aggregate_load) - 1

        if not terminated:
            # Update the current step
            self.current_step += 1

        return self._get_obs(), reward, terminated, False, {}

    # TODO: optional _get_info(self) method

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

    def _g_reward(self, s_t_datetime:datetime, s_t_plus_1_datetime:datetime, power_kw: float) -> float:
        """
        Calculate the reward based on elapsed time, power used, and cost.
        Args:
            s_t_datetime (datetime): Current timestamp.
            s_t_plus_1_datetime (datetime): Next timestamp.
            power_kw (float): Power used in kW.
        Returns:
            float: Calculated reward.
        """

        # Reward is negative of cost incurred
        return self._get_weighted_electricity_cost(s_t_datetime, s_t_plus_1_datetime) * power_kw
    
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

    
    def _f_reward(self, y_t: float, z_t: float, y_t_plus_1: float) -> float:
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
                h_network_input[:, i] = self.h_network_stdscaler.transform(h_network_input[:, i].reshape(-1, 1)).flatten()

        if h_network_target.shape[0] == 1:  # if there is single sample
            h_network_target = self.h_network_stdscaler.transform(h_network_target.reshape(1, 1)).flatten()
        else:
            h_network_target = self.h_network_stdscaler.transform(h_network_target.reshape(-1, 1)).flatten()

        # Convert the buffer to a tensor for H-network input
        h_network_input = torch.tensor(h_network_input).unsqueeze(0)
        h_network_target = torch.tensor(h_network_target).unsqueeze(0)

        # Get the H-network reward
        with torch.no_grad():
            mean = self.h_network(h_network_input)

            loss = self.h_network_criterion(mean, h_network_target, var=torch.ones_like(mean))  # assuming unit variance
            
        # Return the negative loss (as small is better) as the reward
        return -loss.item()
            

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


    # TODO: close(self) method for closing any open resources used by the env