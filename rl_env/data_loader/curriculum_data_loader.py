"""
Curriculum-based Smart Meter Data Loader.

This module implements a sophisticated data loader that supports curriculum learning
with variable episode lengths, gradually introducing longer episodes as training progresses.
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import timedelta, datetime
from typing import List, Tuple, Dict, Callable, Optional
from pathlib import Path
import json

from .base_data_loader import BaseSmartMeterDataLoader


class CurriculumSmartMeterDataLoader(BaseSmartMeterDataLoader):
    """
    Advanced data loader that supports curriculum learning with variable episode lengths.
    
    This loader generates episodes of different lengths (1-5 days) and gradually increases
    the probability of sampling longer episodes as training progresses, enabling more
    stable reinforcement learning.
    """
    
    def __init__(self, 
                 aggregate_load_segments: List[Tuple[datetime, datetime]], 
                 aggregate_load_df: pd.DataFrame,
                 max_episode_days: int = 5,
                 curriculum_schedule: Optional[Callable[[int], Dict[int, float]]] = None,
                 registry_path: Optional[Path] = None):
        """
        Initialize the curriculum data loader.
        
        Args:
            aggregate_load_segments: List of (start, end) datetime tuples
            aggregate_load_df: DataFrame containing the full load data
            max_episode_days: Maximum episode length in days (default: 5)
            curriculum_schedule: Function that takes timestep and returns probability distribution
            registry_path: Path to save episode registry for tracking
        """
        # Initialize base class with registry
        super().__init__(registry_path)
        
        self.aggregate_load_segments = aggregate_load_segments
        self.aggregate_load_df = aggregate_load_df
        self.max_episode_days = max_episode_days
        
        # Default curriculum schedule if none provided
        if curriculum_schedule is None:
            self.curriculum_schedule = self._default_curriculum_schedule
        else:
            self.curriculum_schedule = curriculum_schedule
        
        # Generate episodes of different lengths
        self.episode_segments_by_length = self._generate_variable_length_episodes()
        
        print(f"[CurriculumDataLoader] Generated episodes by length:")
        for length, episodes in self.episode_segments_by_length.items():
            print(f"  {length} day(s): {len(episodes)} episodes")
        
    def _default_curriculum_schedule(self, timestep: int) -> Dict[int, float]:
        """
        Default curriculum schedule that gradually increases probability of longer episodes.
        
        The schedule is designed to:
        1. Start with only 1-day episodes for initial learning
        2. Gradually introduce longer episodes as training progresses
        3. End with a balanced distribution across all episode lengths
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Dictionary mapping episode length (in days) to sampling probability
        """
        # Define curriculum phases (in timesteps)
        # note that these values will be on effect after an episode.
        # due to the .step() -> .reset() -> env calling sample_episode_index() -> UpdateGlobalTimestep() -> update env internal sekf.training_timestep()
        # hence an immediate sampling beyond the phrase boundary will not occur
        # a suggested curriculum scheduling is to set phrase_n = your_targetted_timesteps - an epiosode length
        # e.g. if you want to start sampling 2-day episodes after 100k steps, set phase_1 = 100000 - 24 * 60 (1 min sampling)
        phase_1 = 100000   # First 100k steps: only 1-day episodes
        phase_2 = 300000   # Next 200k steps: introduce 2-day episodes
        phase_3 = 600000   # Next 300k steps: introduce 3-day episodes
        phase_4 = 1000000  # Next 400k steps: introduce 4-day episodes
        # After phase_4: all episode lengths available
        
        if timestep < phase_1:
            return {1: 1.0}
        elif timestep < phase_2:
            progress = (timestep - phase_1) / (phase_2 - phase_1)
            return {
                1: 1.0 - 0.3 * progress,
                2: 0.3 * progress
            }
        elif timestep < phase_3:
            progress = (timestep - phase_2) / (phase_3 - phase_2)
            return {
                1: 0.7 - 0.2 * progress,
                2: 0.3 - 0.05 * progress,
                3: 0.25 * progress
            }
        elif timestep < phase_4:
            progress = (timestep - phase_3) / (phase_4 - phase_3)
            return {
                1: 0.5 - 0.1 * progress,
                2: 0.25 - 0.05 * progress,
                3: 0.25 - 0.05 * progress,
                4: 0.2 * progress
            }
        else:
            # Final distribution - balanced across all lengths
            if self.max_episode_days >= 5:
                return {1: 0.4, 2: 0.2, 3: 0.2, 4: 0.1, 5: 0.1}
            elif self.max_episode_days >= 4:
                return {1: 0.5, 2: 0.25, 3: 0.15, 4: 0.1}
            elif self.max_episode_days >= 3:
                return {1: 0.6, 2: 0.25, 3: 0.15}
            elif self.max_episode_days >= 2:
                return {1: 0.7, 2: 0.3}
            else:
                return {1: 1.0}
    
    def _generate_variable_length_episodes(self) -> Dict[int, List[Tuple[datetime, datetime]]]:
        """
        Generate episodes of different lengths (1-max_episode_days days) from the available segments.
        
        Returns:
            Dictionary mapping episode length to list of (start, end) tuples
        """
        episodes_by_length = {i: [] for i in range(1, self.max_episode_days + 1)}
        
        # Sort segments by start time for continuity checking
        sorted_segments = sorted(self.aggregate_load_segments, key=lambda x: x[0])
        
        for length_days in range(1, self.max_episode_days + 1):
            target_duration = timedelta(days=length_days)
            
            i = 0
            while i < len(sorted_segments):
                start_time = sorted_segments[i][0]
                current_end = sorted_segments[i][1]
                
                # Try to extend the episode to target duration
                j = i + 1
                while j < len(sorted_segments) and (current_end - start_time) < target_duration:
                    next_start = sorted_segments[j][0]
                    next_end = sorted_segments[j][1]
                    
                    # Check if next segment is continuous (within reasonable gap)
                    gap = next_start - current_end
                    if gap <= timedelta(hours=1):  # Allow small gaps
                        current_end = next_end
                        j += 1
                    else:
                        break
                
                # Check if we achieved the target duration
                actual_duration = current_end - start_time
                if actual_duration >= target_duration - timedelta(hours=1):  # Allow some tolerance
                    # Truncate to exact target duration if needed
                    episode_end = min(current_end, start_time + target_duration)
                    episodes_by_length[length_days].append((start_time, episode_end))
                
                # Move to next starting point
                i = j if j > i + 1 else i + 1
        
        return episodes_by_length
    
    def _get_default_registry_path(self) -> str:
        """Return default registry file path for curriculum data loader."""
        return "curriculum_episode_registry.json"
    
    def get_divided_segments_length(self) -> int:
        """Returns the total number of available episodes across all lengths."""
        return sum(len(episodes) for episodes in self.episode_segments_by_length.values())
    
    def sample_episode_index(self, np_random: np.random.Generator, timestep: Optional[int] = None) -> int:
        """
        Sample an episode index based on curriculum schedule.
        
        Args:
            np_random (np.random.Generator): Random number generator for reproducibility in gym environment.
            timestep: Current training timestep for curriculum scheduling
            
        Returns:
            Global episode index
            
        Raises:
            IndexError: If no valid episodes are available
        """
        if timestep is None:
            timestep = 0
        
        # Get curriculum probabilities
        length_probs = self.curriculum_schedule(timestep)
        
        # Filter available lengths and normalize probabilities
        available_lengths = [length for length in length_probs.keys() 
                           if length in self.episode_segments_by_length and 
                           len(self.episode_segments_by_length[length]) > 0]
        
        if not available_lengths:
            # Fallback to 1-day episodes
            available_lengths = [1]
            if len(self.episode_segments_by_length.get(1, [])) == 0:
                raise IndexError("No episodes available for sampling")
        
        # Normalize probabilities for available lengths
        total_prob = sum(length_probs.get(length, 0) for length in available_lengths)
        if total_prob == 0:
            # Equal probability fallback
            normalized_probs = [1.0 / len(available_lengths)] * len(available_lengths)
        else:
            normalized_probs = [length_probs.get(length, 0) / total_prob for length in available_lengths]
        
        # Sample episode length
        chosen_length = np_random.choice(available_lengths, p=normalized_probs)
        
        # Sample specific episode of chosen length
        length_episodes = self.episode_segments_by_length[chosen_length]
        if not length_episodes:
            raise IndexError(f"No episodes available for length {chosen_length}")
            
        episode_idx_within_length = np_random.integers(0, len(length_episodes))
        
        # Convert to global index
        global_idx = 0
        for length in range(1, chosen_length):
            if length in self.episode_segments_by_length:
                global_idx += len(self.episode_segments_by_length[length])
        global_idx += episode_idx_within_length
        
        return global_idx
    
    def get_aggregate_load_segment(self, index: int) -> pd.DataFrame:
        """
        Retrieve a specific episode by global index.
        
        Args:
            index: Global episode index
            
        Returns:
            DataFrame containing the episode data
            
        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self.get_divided_segments_length():
            raise IndexError(f"Episode index {index} out of bounds.")
        
        # Find which length category this index belongs to
        current_idx = 0
        for length in range(1, self.max_episode_days + 1):
            if length not in self.episode_segments_by_length:
                continue
            length_episodes = self.episode_segments_by_length[length]
            if current_idx + len(length_episodes) > index:
                # Found the right category
                local_idx = index - current_idx
                start_datetime, end_datetime = length_episodes[local_idx]
                episode_length = length
                break
            current_idx += len(length_episodes)
        else:
            raise IndexError(f"Could not find episode for index {index}")
        
        # Extract data for the episode
        mask = self.aggregate_load_df['datetime'].between(start_datetime, end_datetime)
        episode_data = self.aggregate_load_df[mask].copy()
        
        # Ensure we have the required columns
        if 'timestamp' not in episode_data.columns or 'aggregate' not in episode_data.columns:
            raise ValueError("Episode data must contain 'timestamp' and 'aggregate' columns")
        
        # Add required columns
        episode_data = episode_data[['timestamp', 'aggregate', 'datetime']].copy()
        episode_data['segment_index'] = index
        
        # Register this episode in the registry and get content ID
        content_id = self._register_episode(index, start_datetime, end_datetime, len(episode_data), episode_length)
        
        # Add episode identification columns
        episode_data['episode_content_id'] = content_id
        episode_data['episode_length_days'] = episode_length
        
        return episode_data
    
    def _register_episode(self, index: int, start_time: datetime, end_time: datetime, 
                         length: int, episode_days: int) -> str:
        """Register episode in the tracking registry and return content ID."""
        # Generate content-based ID
        content_string = f"{start_time}_{end_time}_{length}_{episode_days}"
        content_id = hashlib.md5(content_string.encode()).hexdigest()[:12]
        
        if content_id not in self.registry["episodes"]:
            self.registry["episodes"][content_id] = {
                "content_id": content_id,
                "datetime_range": (start_time.isoformat(), end_time.isoformat()),
                "length": length,
                "episode_days": episode_days,
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
    
    def get_episode_metadata(self, index: int) -> dict:
        """Get metadata for a specific episode."""
        # This would require finding the episode first, which involves the same logic
        # as get_aggregate_load_segment. For now, return basic info.
        return {
            "global_index": index,
            "total_episodes": self.get_divided_segments_length()
        }
    
    def get_curriculum_info(self) -> dict:
        """Get information about the current curriculum state."""
        return {
            "curriculum_enabled": True,
            "max_episode_days": self.max_episode_days,
            "episodes_by_length": {length: len(episodes) 
                                  for length, episodes in self.episode_segments_by_length.items()},
            "total_episodes": self.get_divided_segments_length(),
            "registry_path": str(self.registry_path)
        }
    
    def get_current_curriculum_probs(self, timestep: int) -> Dict[int, float]:
        """Get the current curriculum probabilities for debugging."""
        return self.curriculum_schedule(timestep)
