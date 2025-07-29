"""
Simple Smart Meter Data Loader.

This module implements a basic data loader that provides uniform episode sampling
for the Smart Meter reinforcement learning environment. This is the original
implementation that splits larger segments into fixed-length episodes.
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import timedelta, datetime
from typing import Optional
from pathlib import Path
import json
from .base_data_loader import BaseSmartMeterDataLoader


class SimpleSmartMeterDataLoader(BaseSmartMeterDataLoader):
    """
    Simple data loader that provides uniform episode sampling.
    
    This loader divides the input segments into fixed-length episodes (default 24 hours)
    and provides uniform random sampling without curriculum learning.
    """
    
    def __init__(self, aggregate_load_segments: list, 
                 aggregate_load_df: pd.DataFrame, 
                 segment_length: int = 24,
                 registry_path: Optional[Path] = None):
        """
        Initializes the data loader with segments and DataFrame.
        
        Args:
            aggregate_load_segments: List of (start, end) datetime tuples containing load segments.
            aggregate_load_df: Pandas DataFrame containing the full load data.
            segment_length: Length of each segment in hours (default: 24 for 1-day episodes)
            registry_path: Path to save episode registry for tracking
        """
        # Initialize base class with registry
        super().__init__(registry_path)
        
        self.aggregate_load_segments = aggregate_load_segments
        self.aggregate_load_df = aggregate_load_df
        self.segment_length = timedelta(hours=segment_length)

        self.divided_segments = self._divide_segments()
        
        print(f"[SimpleDataLoader] Generated {self.get_divided_segments_length()} episodes (1 day each)")

    def _divide_segments(self) -> np.ndarray:
        """Divide the aggregate load segments into smaller segments of specified length."""
        num_segments = len(self.aggregate_load_segments)
        divided_segments = []

        for i in range(num_segments):
            segment = self.aggregate_load_segments[i]
            start_time = segment[0]
            end_time = segment[1]
            # Create smaller segments within the specified length
            for j in pd.date_range(start=start_time, end=end_time, freq=self.segment_length):
                divided_segments.append([
                    j.to_pydatetime(), 
                    min((j + self.segment_length - timedelta(microseconds=1)).to_pydatetime(), end_time)
                ])

        return np.array(divided_segments)
    
    def get_divided_segments_length(self) -> int:
        """Returns the number of divided segments."""
        return self.divided_segments.shape[0]
    
    def get_aggregate_load_segment(self, index: int) -> pd.DataFrame:
        """
        Retrieves a specific segment by index.

        Args:
            index: Index of the segment to retrieve.
            
        Returns:
            Pandas DataFrame containing the segment data with columns:
            - timestamp: Unix timestamp
            - aggregate: Aggregate load values  
            - datetime: Datetime objects
            - segment_index: Index of this segment
            
        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self.divided_segments.shape[0]:
            raise IndexError("Segment index out of bounds.")
        
        segment = self.divided_segments[index]

        # Based on the segment, get the corresponding aggregate load data
        start_datetime, end_datetime = segment[0], segment[1]
        aggregate_load = self.aggregate_load_df[self.aggregate_load_df['datetime'].between(start_datetime, end_datetime)]

        # We need the timestamp to create datetime related features in the environment
        aggregate_load = aggregate_load[['timestamp', 'aggregate', 'datetime']].copy()

        # We also add a column for the segment index
        aggregate_load['segment_index'] = index
        
        # Register this episode in the registry and get content ID
        content_id = self._register_episode(index, start_datetime, end_datetime, len(aggregate_load))
        
        # Add episode identification columns
        aggregate_load['episode_content_id'] = content_id
        aggregate_load['episode_length_days'] = 1  # Simple loader always uses 1-day episodes

        return aggregate_load
    
    def _get_default_registry_path(self) -> str:
        """Return default registry file path for simple data loader."""
        return "simple_episode_registry.json"
    
    def _register_episode(self, index: int, start_time: datetime, end_time: datetime, length: int) -> str:
        """Register episode in the tracking registry and return content ID."""
        # Generate content-based ID
        content_string = f"{start_time}_{end_time}_{length}_1"  # 1 day for simple loader
        content_id = hashlib.md5(content_string.encode()).hexdigest()[:12]
        
        if content_id not in self.registry["episodes"]:
            self.registry["episodes"][content_id] = {
                "content_id": content_id,
                "datetime_range": (start_time.isoformat(), end_time.isoformat()),
                "length": length,
                "episode_days": 1,  # Always 1 day for simple loader
                "first_seen": datetime.now().isoformat(),
                "access_count": 0
            }
        
        self.registry["episodes"][content_id]["access_count"] += 1
        self.registry["episodes"][content_id]["last_accessed"] = datetime.now().isoformat()
        
        # Save registry periodically (every 100 accesses)
        total_accesses = sum(ep.get("access_count", 0) for ep in self.registry["episodes"].values())
        if total_accesses % 100 == 0:
            self._save_registry()
        
        return content_id
    
    def _generate_episode_content_id(self, start_time: datetime, end_time: datetime, length: int) -> str:
        """Generate content-based ID for episode identification (legacy method)."""
        content_string = f"{start_time}_{end_time}_{length}_1"  # 1 day for simple loader
        return hashlib.md5(content_string.encode()).hexdigest()[:12]
    
    def sample_episode_index(self, np_random: np.random.Generator, timestep: Optional[int] = None) -> int:
        """
        Sample an episode index uniformly (simple implementation).
        
        Args:
            np_random (np.random.Generator): Random number generator for reproducibility in gym environment.
            timestep (Optional[int]): Current training timestep (ignored in simple implementation)
            
        Returns:
            int: Randomly sampled episode index
        """
        return np_random.integers(0, self.get_divided_segments_length())
    
    def get_episode_metadata(self, index: int) -> dict:
        """Get metadata for a specific episode."""
        if index < 0 or index >= self.get_divided_segments_length():
            raise IndexError("Episode index out of bounds.")
            
        return {
            "global_index": index,
            "total_episodes": self.get_divided_segments_length(),
            "episode_length_days": 1,
            "data_loader_type": "simple"
        }
    
    def get_curriculum_info(self) -> dict:
        """Get information about the current curriculum state."""
        return {
            "curriculum_enabled": False,
            "max_episode_days": 1,
            "episodes_by_length": {1: self.get_divided_segments_length()},
            "total_episodes": self.get_divided_segments_length(),
            "registry_path": str(self.registry_path),
            "data_loader_type": "simple"
        }
    

# Keep backward compatibility - alias to the old class name
SmartMeterDataLoader = SimpleSmartMeterDataLoader
