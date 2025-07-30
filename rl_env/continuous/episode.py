"""
Continuous action space implementation of Smart Meter episode.

This module provides the episode functionality specifically designed for
continuous action spaces, inheriting from the base episode functionality.
"""

import pandas as pd
from ..base.episode import SmartMeterEpisodeBase, EpisodeDataProcessor


class SmartMeterEpisode(SmartMeterEpisodeBase):
    """
    Smart Meter episode for continuous action spaces.
    
    This implementation uses standardization-only data processing,
    which is appropriate for continuous action spaces.
    """
    
    def __init__(self, selected_aggregate_load_df: pd.DataFrame, data_processor: EpisodeDataProcessor):
        """
        Initialize the continuous episode.
        
        Args:
            selected_aggregate_load_df: The DataFrame containing aggregate load data
            data_processor: The data processing strategy (should be ContinuousDataProcessor)
        """
        super().__init__(selected_aggregate_load_df, data_processor)
