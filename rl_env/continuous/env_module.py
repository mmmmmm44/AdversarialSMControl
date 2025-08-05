"""
Continuous action space Smart Meter RL environment implementation.

This module implements the Smart Meter RL environment for continuous action spaces,
extending the base environment with continuous-specific functionality.
"""

from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import gymnasium as gym
import pandas as pd
from typing import Dict, Tuple, Optional
import math

from rl_env.training_mode import TrainingMode
from rl_env.base.env_module import SmartMeterEnvironmentBase
from rl_env.base.battery import BatteryFactory
from rl_env.base.episode import EpisodeFactory
from utils import print_log


class SmartMeterContinuousEnv(SmartMeterEnvironmentBase):
    """
    Smart Meter World Environment for Reinforcement Learning with continuous action space.
    This environment simulates an energy management unit connected with a rechargeable battery.
    The charging/discharging action of the battery is continuous.
    """

    def _init_environment_specifics(self, rb_config: Optional[dict]):
        """Initialize continuous-specific components."""

        # Create episode using factory
        self.episode = EpisodeFactory.create(
            'continuous', 
            self.smart_meter_data_loader.get_aggregate_load_segment(self.selected_idx)
        )
        
        # Battery configuration
        self.init_soc = 0.15 if rb_config is None else rb_config.get('init_soc', 0.5)
        
        # Create battery using factory
        default_battery_config = {
            'capacity': 8.0,  # kWh
            'max_charging_rate': 4.0,  # kW
            'efficiency': 1.0,  # 100% efficiency
            'init_soc': self.init_soc
        }
        
        battery_config = {**default_battery_config, **(rb_config or {})}
        self.battery = BatteryFactory.create('continuous', **battery_config)
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "aggregate_load": gym.spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32),    # standardized aggregate load
            "battery_soc": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),       # normalized battery state of charge
            "timestamp_features": gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)  # Hour, Day of Week, Month
        })

        # Define action space  
        self.low_level_action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # Set the main action space to be the same as low level action space
        self.action_space = self.low_level_action_space

    def _get_obs(self) -> Dict:
        """Get observation for the current step."""

        current_step = self.episode.get_current_step()

        soc = self.battery.get_normalized_state_of_charge()
        current_load = self.episode.df.iloc[current_step]['aggregate_std']  # receive standardized aggregate load
        
        def _create_timestamp_features(timestamp: int) -> np.ndarray:
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

        timestamp = self.episode.df.iloc[current_step]['timestamp']
        timestamp_features = _create_timestamp_features(timestamp)
        return {
            "aggregate_load": np.array([current_load], dtype=np.float32),
            "battery_soc": np.array([soc], dtype=np.float32),
            "timestamp_features": timestamp_features
        }

    def reset(self, episode_idx: int = None, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Start a new episode - continuous environment logic."""
        super().reset(seed=seed, options=options)

        # Reset H-network episode inference buffer
        self.h_network_episode_inference_buffer.clear()

        # Reset battery
        self.battery.reset(self.init_soc)

        # reset the episodes rewards
        self.per_episode_rewards = []
        
        # select an episode to reset with
        # if episode_idx is None, use curriculum sampling (if available) or random selection
        # if episode_idx is provided, check if it is valid and use it
        if episode_idx is not None and (episode_idx < 0 or episode_idx >= self.smart_meter_data_loader.get_divided_segments_length()):
            raise ValueError(f"Invalid episode index: {episode_idx}. Must be between 0 and {self.smart_meter_data_loader.get_divided_segments_length() - 1}.")

        # Use curriculum sampling if no specific episode is requested
        # TODO
        # note that if using vectorized environments, the self.training_timestep should NOT be shared across environments
        # Hence, the last reference training timestep (when an env ends) should be set by a callback (i.e. we need to implement a custom callback for this)
        if episode_idx is None:
            self.selected_idx = self.smart_meter_data_loader.sample_episode_index(self.np_random, self.training_timestep)
        else:
            self.selected_idx = episode_idx
        
        # Create new episode using factory
        self.episode = EpisodeFactory.create(
            'continuous',
            self.smart_meter_data_loader.get_aggregate_load_segment(self.selected_idx)
        )
        self.episode_info_list = []

        # Get episode metadata for enhanced logging
        episode_metadata = self.smart_meter_data_loader.get_episode_metadata(self.selected_idx)
        episode_length_days = episode_metadata.get('episode_length_days', 1)
        
        print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Resetting environment with episode {self.selected_idx}. Episode info: {self.episode.get_episode_info()}")
        if self.curriculum_info.get('curriculum_enabled', False):
            episode_length_days = self.curriculum_info.get('episodes_by_length', {}).get(str(len(self.episode.df) // (24 * 60)), 0)
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode length: {episode_length_days} day(s), Training timestep: {self.training_timestep}")
        
        self.send_env_info()  # send the environment info to the render window

        obs = self._get_obs()
        info = self._get_info(obs)

        self.episode_info_list.append(info)

        # TODO
        # Send environment info to render window if connected
        # if self.render_mode == "human" and self.render_connected:
        #     env_info = {
        #         "episode_info": self.episode.get_episode_info(),
        #         "battery_config": self.battery.get_battery_config(),
        #         "mode": str(self.mode),
        #         "reward_lambda": self.reward_lambda
        #     }
        #     self.send_env_info_to_render_window(env_info)

        return obs, info
    
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
        info_dict.update({"f_signal" + "-" + k : v for k, v in f_signal_additional_info.items()} if f_signal_additional_info is not None else {})

        return info_dict

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        Args:
            action: Action taken by the agent (charging or discharging power).
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        current_step = self.episode.get_current_step()
        
        y_t = self.episode.df.iloc[current_step]['aggregate']  # actual load in W

        # Extract action (continuous value between -1 and 1)
        power_kw_normalized = action[0]
        
        # Get current and next datetime
        s_t_datetime = self.episode.df.iloc[current_step]['datetime']
        s_t_plus_1_datetime = self.episode.df.iloc[current_step + 1]['datetime'] if current_step + 1 < self.episode.get_episode_length() else s_t_datetime + pd.Timedelta(minutes=1)

        # Calculate duration
        duration = int((s_t_plus_1_datetime - s_t_datetime).total_seconds())
        
        # Update battery SoC in the episode dataframe
        self.episode.df.iat[current_step, self.episode.df.columns.get_loc('battery_soc')] = self.battery.get_normalized_state_of_charge()
        
        # Calculate unnormalized power from normalized action
        power_kw = self.battery.compute_unnormalized_charge(float(power_kw_normalized))
        power_charged_discharged = self.battery.charge_discharge(power_kw, duration, y_t)

        # --------------------
        # after applying the action
        # advance to the next step, but we have to calculate the grid load
        # hence y_{t} is the aggregated user load, and we compare whether our battery action can help mask the user load y_{t+1}, given previous (masked) grid load z_{t} (an exposed information that may be acquired by attacker)
        # --------------------
        
        # Calculate grid load
        z_t = y_t + power_charged_discharged * 1000
        z_t = np.clip(z_t, 0, None)  # ensure grid load is non-negative
        
        # Update grid load in the episode dataframe
        self.episode.df.iat[current_step, self.episode.df.columns.get_loc('grid_load')] = z_t
        
        # --------------------
        # reward component calculation
        # --------------------

        # For privacy signal, we need y_t+1 if available
        if current_step + 1 < len(self.episode.df):
            y_t_plus_1 = self.episode.df.iloc[current_step + 1]['aggregate']
        else:
            y_t_plus_1 = y_t  # use current load as fallback
            
        g_signal = self._g_signal(
            s_t_datetime=s_t_datetime,
            s_t_plus_1_datetime=s_t_plus_1_datetime,
            power_kw=power_charged_discharged
        )

        f_signal, f_signal_additional_info = self._f_signal(
            y_t=y_t,
            z_t=z_t,
            y_t_plus_1=y_t_plus_1
        )

        # Compute battery difference signal
        battery_difference_signal = self._ep_battery_difference_signal()

        # --------------------
        # compute reward & handle transition
        # --------------------

        # Move to next step
        self.episode.set_current_step(current_step + 1)

        # determine termination condition
        terminated = self.episode.get_current_step() >= self.episode.get_episode_length() - 1     # -1 because we want to stop before the last step to avoid index out of range
        truncated = False
        
        # Calculate reward
        reward = -(self.reward_lambda * g_signal + (1 - self.reward_lambda) * f_signal)

        # ignore that, as we don't have time to test this
        # reward -= battery_difference_signal if terminated else 0.0

        
        next_obs = self._get_obs()
        next_info = self._get_info(
            obs=next_obs, power_kw=power_kw, power_charged_discharged=power_charged_discharged, reward=reward, f_signal=f_signal, g_signal=g_signal, f_signal_additional_info=f_signal_additional_info
        )

        # logging per-step information of this episode
        self.episode_info_list.append(next_info)
        self.per_episode_rewards.append({
            "reward": float(reward),
            "f_signal": float(f_signal),
            "g_signal": float(g_signal),
        })

        # log the episode when terminated
        if terminated:
            # push the old episode to the HNetworkRLModule buffer when only in training mode
            if self.mode.value == TrainingMode.TRAIN.value:
                self.h_network_rl_module.push_to_replay_buffer(self.episode)

            # calculate the sum of rewards for the episode
            episode_sum = math.fsum([item["reward"] for item in self.per_episode_rewards])
            episode_reward_stats = {
                "reward_sum": float(episode_sum),
                "reward_mean": float(np.mean([item["reward"] for item in self.per_episode_rewards])),
                "reward_std": float(np.std([item["reward"] for item in self.per_episode_rewards])),
                "f_signal_sum": float(np.sum([item["f_signal"] for item in self.per_episode_rewards])),
                "f_signal_mean": float(np.mean([item["f_signal"] for item in self.per_episode_rewards])),
                "f_signal_std": float(np.std([item["f_signal"] for item in self.per_episode_rewards])),
                "g_signal_sum": float(np.sum([item["g_signal"] for item in self.per_episode_rewards])),
                "g_signal_mean": float(np.mean([item["g_signal"] for item in self.per_episode_rewards])),
                "g_signal_std": float(np.std([item["g_signal"] for item in self.per_episode_rewards])),
            }
            self.prev_episode_reward_stats = episode_reward_stats  # store the previous episode reward stats for logging
            

            self.per_episode_rewards = []  # reset the per-episode rewards for the next episode

            # get the episode info with metadata
            # this will be handled by the callback
            self.prev_episode_info_with_metadata = self._get_episode_info(self.episode_info_list)

            # Leave all the saving logic to the callback, for synchronization with number of episodes
            # when we create a DummyVecEnv with n_envs > 1

            # if self.mode.value == TrainingMode.TRAIN.value:
            #     self._save_prev_episode_info_list(self.log_folder, len(self.episodes_rewards))
            # else:
            #     self._save_prev_episode_info_list(self.log_folder, self.selected_idx)

            # if self.mode.value == TrainingMode.TRAIN.value:
            #     self._save_episode_df(self.log_folder, len(self.episodes_rewards))  # save the episode DataFrame, name the file based on the number of episodes trained so far
            # else:
            #     self._save_episode_df(self.log_folder, self.selected_idx)       # name the file based on episode index (in the validation and test datasets)

            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode finished. Sum of rewards: {episode_sum}. Mean of rewards: {episode_reward_stats['reward_mean']}. Std of rewards: {episode_reward_stats['reward_std']}")
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode f_signal sum: {episode_reward_stats['f_signal_sum']}. Mean: {episode_reward_stats['f_signal_mean']}. Std: {episode_reward_stats['f_signal_std']}")
            print_log(f"[{self.__class__.__name__} {str(self.mode).capitalize()}] Episode g_signal sum: {episode_reward_stats['g_signal_sum']}. Mean: {episode_reward_stats['g_signal_mean']}. Std: {episode_reward_stats['g_signal_std']}")
        return next_obs, reward, terminated, truncated, next_info

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
    
    def _g_signal(self, s_t_datetime: datetime, s_t_plus_1_datetime: datetime, power_kw: float) -> float:
        """
        Calculate the g_signal (cost signal) for the reward function - Implemented by concrete classes.

        Args:
            s_t_datetime (datetime): Start datetime.
            s_t_plus_1_datetime (datetime): End datetime.
            power_kw (float): Power consumption in kW.
            
        Returns:
            float: g signal value.
        """
        # return cost incurred for the power used in the time period
        return SmartMeterEnvironmentBase._get_weighted_electricity_cost(s_t_datetime, s_t_plus_1_datetime) * abs(power_kw) \
            + SmartMeterEnvironmentBase._get_standing_charge_per_day() * ((s_t_plus_1_datetime - s_t_datetime).total_seconds() / (24 * 60 * 60))  # convert seconds to days

    def _ep_battery_difference_signal(self):
        """
        Check if the battery level returns to the initial state after one episode.
        This should encourage the agent to develop a charging-at-night and discharging-in-day strategy.
        """
        
        final_soc = self.battery.get_normalized_state_of_charge()

        return abs(final_soc - self.init_soc)