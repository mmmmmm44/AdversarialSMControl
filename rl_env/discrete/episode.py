"""
Discrete action space implementation of Smart Meter episode.

This module provides the episode functionality specifically designed for
discrete action spaces, inheriting from the base episode functionality.
"""

import pandas as pd
from ..base.episode import SmartMeterEpisodeBase, EpisodeDataProcessor


class SmartMeterDiscreteEpisode(SmartMeterEpisodeBase):
    """
    Smart Meter episode for discrete action spaces.
    
    This implementation uses logit conversion data processing,
    which discretizes the aggregate load for discrete action spaces.
    """
    
    def __init__(self, selected_aggregate_load_df: pd.DataFrame, data_processor: EpisodeDataProcessor):
        """
        Initialize the discrete episode.
        
        Args:
            selected_aggregate_load_df: The DataFrame containing aggregate load data
            data_processor: The data processing strategy (should be DiscreteDataProcessor)
        """
        super().__init__(selected_aggregate_load_df, data_processor)
    
    def convert_load_to_logit(self, step_size: int = 50):
        """
        Convert the aggregate load values to logits for discrete action space.
        
        This method is kept for backward compatibility, but the actual processing
        is now handled by the DiscreteDataProcessor strategy.
        
        Args:
            step_size: The step size for discretization (default: 50W)
        """
        # This functionality is now handled by DiscreteDataProcessor
        # Keeping this method for backward compatibility
        self.df['aggregate_logit'] = (self.df['aggregate'] / step_size).round().astype(int)
