"""
Abstract base class for Smart Meter episode implementations.

This module provides the base episode functionality that is common between
continuous and discrete action space implementations, using the Strategy Pattern
to handle data processing differences.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any


class EpisodeDataProcessor(ABC):
    """
    Abstract strategy for processing episode data.
    
    This encapsulates the different data processing approaches needed
    for continuous vs discrete action spaces.
    """
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the episode data according to the specific strategy.
        
        Args:
            df: The DataFrame to process
            
        Returns:
            The processed DataFrame
        """
        pass
    
    @abstractmethod
    def should_reprocess_on_reset(self) -> bool:
        """
        Whether data should be reprocessed when the episode is reset.
        
        Returns:
            True if reprocessing is needed, False otherwise
        """
        pass


class ContinuousDataProcessor(EpisodeDataProcessor):
    """
    Data processor for continuous action spaces.
    
    Only performs standardization, no discretization needed.
    """
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for continuous action space (standardization only)."""
        # No additional processing needed for continuous
        return df
    
    def should_reprocess_on_reset(self) -> bool:
        """Continuous processing should be redone on reset."""
        return True


class DiscreteDataProcessor(EpisodeDataProcessor):
    """
    Data processor for discrete action spaces.
    
    Performs logit conversion for discretization in addition to standardization.
    """
    
    def __init__(self, step_size: int = 50):
        """
        Initialize the discrete data processor.
        
        Args:
            step_size: The step size for discretization (default: 50W)
        """
        self.step_size = step_size
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for discrete action space (includes logit conversion)."""
        df_processed = df.copy()
        
        # Convert aggregate load values to logits for discrete action space
        df_processed['aggregate_logit'] = (df_processed['aggregate'] / self.step_size).round().astype(int)
        
        return df_processed
    
    def should_reprocess_on_reset(self) -> bool:
        """Discrete processing doesn't need to be redone on reset."""
        return False


class SmartMeterEpisodeBase(ABC):
    """
    Abstract base class for Smart Meter episodes.
    
    Contains the common functionality shared between continuous and discrete
    episode implementations, using the Strategy Pattern for data processing.
    """
    
    def __init__(self, selected_aggregate_load_df: pd.DataFrame, data_processor: EpisodeDataProcessor):
        """
        Initialize the episode with data processing strategy.
        
        Args:
            selected_aggregate_load_df: The DataFrame containing aggregate load data
            data_processor: The strategy for processing episode data
        """
        self.data_processor = data_processor
        self.df = selected_aggregate_load_df.copy()
        self.df['grid_load'] = None  # add a new column for grid load
        self.df['battery_soc'] = None  # add a new column for battery state of charge
        
        self.current_step = 0  # Current step in the episode
        
        # Process data according to strategy
        self.df = self.data_processor.process_data(self.df)
        
        # Always perform standardization (after strategy-specific processing)
        self.standard_scalar = StandardScaler()
        self.standardize_aggregate_load()
    
    def reset(self, selected_aggregate_load_df: pd.DataFrame):
        """
        Reset the episode with a new aggregate load DataFrame.
        
        Args:
            selected_aggregate_load_df: The DataFrame containing aggregate load data for the episode
        """
        self.df = selected_aggregate_load_df.copy()
        self.df['grid_load'] = None
        self.df['battery_soc'] = None
        
        self.current_step = 0
        
        # Process data according to strategy
        self.df = self.data_processor.process_data(self.df)
        
        # Re-standardize if the strategy requires it
        if self.data_processor.should_reprocess_on_reset():
            self.standard_scalar = StandardScaler()
            self.standardize_aggregate_load()
    
    def get_current_step(self) -> int:
        """
        Get the current step in the episode.
        
        Returns:
            The current step index
        """
        return self.current_step
    
    def set_current_step(self, step: int):
        """
        Set the current step in the episode.
        
        Args:
            step: The step index to set
            
        Raises:
            ValueError: If step index is out of bounds
        """
        if step < 0 or step >= len(self.df):
            raise ValueError("Step index out of bounds.")
        self.current_step = step
        
    def get_episode_length(self) -> int:
        """
        Get the length of the episode.
        
        Returns:
            The number of steps in the episode
        """
        return len(self.df)
    
    def get_episode_info(self) -> Dict[str, Any]:
        """
        Get information about the episode.
        
        Returns:
            A dictionary containing episode information such as length and current step
        """
        return {
            "length": self.get_episode_length(),
            "datetime_range": (self.df['datetime'].min(), self.df['datetime'].max()),
        }
    
    def standardize_aggregate_load(self):
        """
        Standardize the aggregate load data in the episode.
        
        Raises:
            ValueError: If aggregate load data is not found in the DataFrame
        """
        if 'aggregate' not in self.df.columns:
            raise ValueError("Aggregate load data not found in the DataFrame.")
        
        self.df['aggregate_std'] = self.standard_scalar.fit_transform(self.df[['aggregate']].values).flatten()

    def save_df(self, file_path: str):
        """
        Save the episode DataFrame to a pkl file.
        
        Args:
            file_path: The path to the pkl file
        """
        self.df.to_pickle(file_path)


class EpisodeFactory:
    """
    Factory for creating different types of Smart Meter episodes.
    
    Encapsulates the creation logic and strategy selection for episodes.
    """
    
    @staticmethod
    def create(action_type: str, selected_aggregate_load_df: pd.DataFrame, **kwargs) -> SmartMeterEpisodeBase:
        """
        Create a Smart Meter episode appropriate for the given action type.
        
        Args:
            action_type: The type of action space ('continuous' or 'discrete')
            selected_aggregate_load_df: The DataFrame containing aggregate load data
            **kwargs: Additional arguments specific to the episode type
                For discrete: step_size (int, default=50)
            
        Returns:
            A SmartMeterEpisodeBase instance appropriate for the action type
            
        Raises:
            ValueError: If action_type is not supported
        """
        if action_type == 'continuous':
            from ..continuous.episode import SmartMeterEpisode
            processor = ContinuousDataProcessor()
            return SmartMeterEpisode(selected_aggregate_load_df, processor)
        elif action_type == 'discrete':
            from ..discrete.episode import SmartMeterDiscreteEpisode
            step_size = kwargs.get('step_size', 50)
            processor = DiscreteDataProcessor(step_size=step_size)
            return SmartMeterDiscreteEpisode(selected_aggregate_load_df, processor)
        else:
            raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")
