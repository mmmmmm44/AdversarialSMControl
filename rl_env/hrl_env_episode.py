import pandas as pd
import numpy as np
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

        self.standard_scalar = StandardScaler()      # per-sequence standard scaler, similar to what it was done in NeuralNILM
        self.standardize_aggregate_load()            # standardize the aggregate load data

        self.current_step = 0  # Current step in the episode

    def reset(self, selected_aggregate_load_df: pd.DataFrame):
        """
        Reset the episode with a new aggregate load DataFrame.
        Args:
            selected_aggregate_load_df (pd.DataFrame): The DataFrame containing aggregate load data for the episode.
        """

        self.df = selected_aggregate_load_df.copy()
        self.df['grid_load'] = None
        self.df['battery_soc'] = None

        # standardize the aggregate load
        self.standard_scalar = StandardScaler()
        self.standardize_aggregate_load()

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
    
    def standardize_aggregate_load(self):
        """
        Standardize the aggregate load data in the episode.
        """
        if 'aggregate' not in self.df.columns:
            raise ValueError("Aggregate load data not found in the DataFrame.")
        
        self.df['aggregate_std'] = self.standard_scalar.fit_transform(self.df[['aggregate']].values).flatten()

    def save_df(self, file_path: str):
        """
        Save the episode DataFrame to a pkl file.
        Args:
            file_path (str): The path to the pkl file.
        """
        self.df.to_pickle(file_path)