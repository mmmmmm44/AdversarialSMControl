"""
Base data loader interface for Smart Meter data.

This module provides the abstract base class that defines the interface
for all Smart Meter data loaders, enabling different implementations
including simple episode sampling and curriculum learning approaches.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


class BaseSmartMeterDataLoader(ABC):
    """
    Abstract base class for Smart Meter data loaders.
    
    This interface defines the common methods that all data loaders
    must implement to work with the SmartMeterWorld environment.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the base data loader with registry functionality.
        
        Args:
            registry_path: Optional path for episode registry file
        """
        self.registry_path = registry_path or Path(self._get_default_registry_path())
        self._init_episode_registry()
    
    @abstractmethod
    def get_divided_segments_length(self) -> int:
        """
        Returns the number of available segments/episodes.
        
        Returns:
            int: Total number of episodes available for sampling
        """
        pass
    
    @abstractmethod
    def get_aggregate_load_segment(self, index: int) -> pd.DataFrame:
        """
        Retrieves a specific segment/episode by index.
        
        Args:
            index (int): Index of the segment to retrieve
            
        Returns:
            pd.DataFrame: DataFrame containing the episode data with required columns:
                - timestamp: Unix timestamp
                - aggregate: Aggregate load values
                - datetime: Datetime objects
                - segment_index: Index of this segment
                - episode_content_id: Content-based unique identifier
                - episode_length_days: Episode length in days
                
        Raises:
            IndexError: If index is out of bounds
        """
        pass
    
    @abstractmethod
    def sample_episode_index(self, np_random: np.random.Generator, timestep: Optional[int] = None) -> int:
        """
        Sample an episode index, potentially based on curriculum or other logic.
        
        Args:
            np_random (np.random.Generator): Random number generator for reproducibility in gym environment.
            timestep (Optional[int]): Current training timestep for curriculum scheduling.
                                    If None, uses default sampling strategy.
                                    
        Returns:
            int: Index of the sampled episode
            
        Raises:
            IndexError: If no valid episodes are available
        """
        pass
    
    @abstractmethod
    def get_episode_metadata(self, index: int) -> dict:
        """
        Get metadata for a specific episode.
        
        Args:
            index (int): Index of the episode
            
        Returns:
            dict: Metadata dictionary containing episode information
        """
        pass
    
    @abstractmethod
    def get_curriculum_info(self) -> dict:
        """
        Get information about the current curriculum state.
        
        Returns:
            dict: Dictionary containing curriculum information
        """
        pass
    
    @abstractmethod
    def _get_default_registry_path(self) -> str:
        """
        Return default registry file path for this loader type.
        
        Returns:
            str: Default registry file path
        """
        pass
    
    # Common Registry Implementation
    def _init_episode_registry(self):
        """Initialize or load the episode registry for tracking."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "episodes": {},  # episode_content_id -> episode_info
                "metadata": {"created": datetime.now().isoformat()}
            }
    
    def _save_registry(self):
        """Save the episode registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4, default=str)
    
    def get_registry_info(self) -> dict:
        """Get information about the episode registry."""
        return {
            "total_registered_episodes": len(self.registry["episodes"]),
            "registry_metadata": self.registry.get("metadata", {}),
            "registry_path": str(self.registry_path)
        }
    
    def save_registry(self):
        """Manually save the episode registry."""
        self._save_registry()
