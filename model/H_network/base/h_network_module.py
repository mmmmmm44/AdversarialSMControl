"""
Abstract base class for H-Network RL modules.

This module provides the base functionality that is common between
continuous and discrete H-Network RL implementations, using the Template Method Pattern
to handle action-type specific differences in training and signal computation.
"""

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
import json
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ..h_network_arch import HNetworkType
from ..h_network import HNetwork as HNetwork
from ..h_network2 import HNetwork as HNetwork2


class HNetworkRLModuleBase(ABC):
    """
    Abstract base class for H-Network RL modules.
    
    Contains the common functionality shared between continuous and discrete
    H-Network implementations, using the Template Method Pattern for 
    action-type specific operations.
    """
    
    def __init__(self, h_network_type: HNetworkType, device: str, optimizer_class=None, optimizer_kwargs=None):
        """
        Initialize the H-Network RL module base.
        
        Args:
            h_network_type: The type of H-network to use
            device: Device to run the H-network on (e.g., 'cpu' or 'cuda')
            optimizer_class: Optimizer class (default: torch.optim.Adam)
            optimizer_kwargs: Dict of optimizer arguments (default: {'lr': 1e-3})
        """
        self.device = device
        self.h_network_type = h_network_type
        
        # Optimizer configuration
        self.optimizer_class = optimizer_class or torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        
        self.h_network = None                   # Placeholder for H-network, to be loaded or trained
        self.H_NETWORK_MAXSEQLEN = 512          # Maximum sequence length for H-network inference buffer
        
        self.replay_buffer = deque()            # Replay buffer for H-network training, stores SmartMeterEpisode objects
        
        
        # Initialize the action-type specific loss criterion
        self.h_network_criterion = self._create_loss_criterion()
        
        # Logging results - mean and std of training loss for each training step
        self.train_loss_list = []

    
    @abstractmethod
    def _create_loss_criterion(self):
        """Create the loss criterion specific to the action type."""
        pass

    def _create_training_criterion(self):
        """Create the training criterion specific to the action type."""
        pass
    
    @abstractmethod
    def _get_network_output_dim(self, **kwargs) -> int:
        """Get the output dimension for the H-network based on action type."""
        pass
    
    @abstractmethod 
    def _get_supported_h_network_types(self) -> list:
        """Get the list of supported H-network types for this action space."""
        pass
    
    def initialize_h_network(self, **kwargs):
        """
        Initialize the H-network. This replaces any existing H-network with a new one.
        
        Args:
            **kwargs: Additional arguments for network initialization
                For discrete: output_dim (int, default=101)
        
        Returns:
            The initialized H-network
        
        Raises:
            ValueError: If h_network_type is not supported for this action space
        """
        output_dim = self._get_network_output_dim(**kwargs)
        supported_types = self._get_supported_h_network_types()
        
        if self.h_network_type not in supported_types:
            raise ValueError(f"Unsupported H-network type: {self.h_network_type}. "
                           f"Supported types for this action space are: {supported_types}")
        
        if self.h_network_type.value == HNetworkType.H_NETWORK.value:
            network = HNetwork(input_dim=2, hidden_dim=44, output_dim=output_dim)
        elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
            network = HNetwork2(input_dim=2, hidden_dim=44, output_dim=output_dim)
        else:
            raise ValueError(f"Unknown H-network type: {self.h_network_type}")
        
        self.h_network = network.to(self.device)
        return self.h_network
    
    def initialize_h_network_training(self):
        """
        Initialize the H-network training components, including the loss function and optimizer.
        This method should be called after the H-network is initialized.
        """
        if self.h_network is None:
            raise RuntimeError("H-network is not initialized. Please call initialize_h_network() first.")

        # Create training criterion with reduction='mean' for stable training
        self.h_network_training_criterion = self._create_training_criterion()
        
        # Create optimizer with configured class and parameters
        self.h_network_training_optimizer = self.optimizer_class(
            self.h_network.parameters(), 
            **self.optimizer_kwargs
        )

        self.train_loss_list = []  # Reset the training loss list
        self.h_network.to(self.device)  # Move the H-network to the specified device
    
    def set_h_network(self, h_network):
        """Set the H-network instance."""
        self.h_network = h_network
    
    @abstractmethod
    def train(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Train the H-network using the replay buffer.
        
        This is the main method that differs between action types due to different
        data processing requirements.
        
        Returns:
            Tuple of (mean_loss, std_loss) or (None, None) if no training data
        """
        pass
    
    @abstractmethod  
    def compute_f_signal(self, h_network_input: torch.Tensor, h_network_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the f signal using the H-network.
        
        This method differs between action types due to different target processing.
        
        Args:
            h_network_input: Input to the H-network
            h_network_target: Target for the H-network
            
        Returns:
            Tuple of (f_signal, additional_info)
        """
        pass
    
    def _chunk_pad_mask_sequences(self, input_sequences, target_sequences, chunk_size=512, stride=64, padding_value=0.0):
        """
        Chunk, pad, and mask input/target sequences for LSTM training.
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

        Example usage:
            padded_inputs, padded_targets, mask = self._chunk_pad_mask_sequences(input_sequences, target_sequences)
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
    
    def push_to_replay_buffer(self, episode):
        """
        Push an episode to the replay buffer.
        
        Args:
            episode: SmartMeterEpisode object to add to the buffer
        """
        self.replay_buffer.append(episode)
    
    def save_train_loss_list(self, file_path: Path):
        """
        Save the training loss list to a JSON file.
        
        Args:
            file_path: Path to save the loss list
        """
        loss_data = {
            "train_loss_list": self.train_loss_list,
            "num_episodes": len(self.train_loss_list)
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
    
    def save_h_network(self, file_path: Path):
        """
        Save the H-network state dict to a file.
        
        Args:
            file_path: Path to save the H-network
            
        Raises:
            RuntimeError: If h_network has not been initialized
        """
        if self.h_network is None:
            raise RuntimeError("H-network has not been initialized. Cannot save.")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.h_network.state_dict(), file_path)
    
    def load_h_network(self, file_path: Path):
        """
        Load the H-network state dict from a file.
        
        Args:
            file_path: Path to load the H-network from
            
        Raises:
            RuntimeError: If h_network has not been initialized
            FileNotFoundError: If the file does not exist
        """
        if self.h_network is None:
            raise RuntimeError("H-network has not been initialized. Call initialize_h_network() first.")
        
        if not file_path.exists():
            raise FileNotFoundError(f"H-network file not found: {file_path}")
        
        state_dict = torch.load(file_path, map_location=self.device)
        self.h_network.load_state_dict(state_dict)


class HNetworkModuleFactory:
    """
    Factory for creating H-Network RL modules based on action type.
    
    Encapsulates the creation logic and provides a unified interface for
    creating appropriate H-Network modules.
    """
    
    @staticmethod
    def create(action_type: str, h_network_type: HNetworkType, device: str, **kwargs):
        """
        Create an H-Network RL module appropriate for the given action type.
        
        Args:
            action_type: The type of action space ('continuous' or 'discrete')
            h_network_type: The type of H-network architecture to use
            device: Device to run the H-network on
            **kwargs: Additional arguments specific to the action type
                For discrete: output_dim (int, default=101)
            
        Returns:
            HNetworkRLModuleBase instance appropriate for the action type
            
        Raises:
            ValueError: If action_type is not supported
        """
        action_type = action_type.lower().strip()
        
        if action_type == 'continuous':
            from ..continuous.h_network_module import HNetworkRLModule
            return HNetworkRLModule(h_network_type, device)
        elif action_type == 'discrete':
            from ..discrete.h_network_module import HNetworkDiscreteRLModule
            return HNetworkDiscreteRLModule(h_network_type, device, **kwargs)
        else:
            raise ValueError(f"Unsupported action type: {action_type}. Must be 'continuous' or 'discrete'.")
