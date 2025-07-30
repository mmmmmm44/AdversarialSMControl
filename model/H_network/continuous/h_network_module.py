"""
Continuous action space implementation of H-Network RL module.

This module provides the H-Network functionality specifically designed for
continuous action spaces, inheriting from the base H-Network functionality.
"""

from collections import deque
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ..base.h_network_module import HNetworkRLModuleBase
from ..h_network_arch import HNetworkType


class HNetworkRLModule(HNetworkRLModuleBase):
    """
    H-Network RL module for continuous action spaces.
    
    Uses GaussianNLLLoss for privacy signal computation and supports
    both H_NETWORK and H_NETWORK2 architectures.
    """
    
    def __init__(self, h_network_type: HNetworkType, device: str, optimizer_class=None, optimizer_kwargs=None):
        """
        Initialize the continuous H-Network RL module.
        
        Args:
            h_network_type: Type of H-Network architecture
            device: Device to run on ('cpu' or 'cuda')
            optimizer_class: Optimizer class (default: torch.optim.Adam)
            optimizer_kwargs: Dict of optimizer arguments (default: {'lr': 1e-3})
        """
        super().__init__(h_network_type, device, optimizer_class, optimizer_kwargs)
    
    def _create_loss_criterion(self):
        """Create GaussianNLLLoss criterion for continuous action spaces."""
        return nn.GaussianNLLLoss(reduction='none')
    
    def _create_training_criterion(self):
        """Create training criterion with reduction='mean' for stable training."""
        return nn.GaussianNLLLoss()
    
    def _get_network_output_dim(self, **kwargs) -> int:
        """Get output dimension for continuous action space (regression)."""
        return 1
    
    def _get_supported_h_network_types(self) -> list:
        """Get supported H-network types for continuous action space."""
        return [HNetworkType.H_NETWORK, HNetworkType.H_NETWORK2]
    
    def _create_h_network_sequences(self, replay_buffer):
        """
        Create sequences for H-network training from the replay buffer.
        Args:
            replay_buffer: deque, contains tuples of (z_t, y_t)
        Returns:
            input_sequences: list of np.ndarray, each shape (T, 2)
            target_sequences: list of np.ndarray, each shape (T,)
        """
        input_sequences = []
        target_sequences = []

        for seq in replay_buffer:
            # Create a sequence of inputs and targets
            input_seq = seq[:-1, :]
            target_seq = seq[1:, 1]

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        return input_sequences, target_sequences

    def train(self):
        """
        Train the H-network using the replay buffer.
        This method should be called by stable-baselines3 RL callback.

        Note that any print func in this method should be forbidden, since it will break the stable-baseline3 jupyter notebook logging.
        """
        
        _buffer = deque()       # Replay buffer for H-network training, to store recent np.array([(z_{1:T}, y_{1:T})]) for training

        # replay_buffer stores SmartMeterEpisode objects
        # for each episode, we apply standardization to the grid load.
        for episode in self.replay_buffer:
            episode.df['grid_load_std'] = episode.standard_scalar.transform(episode.df[['grid_load']].values).flatten()

            _buffer.append(episode.df[['grid_load_std', 'aggregate_std']].values)  # Store the standardized grid load and aggregate load

        if len(_buffer) == 0:
            # print_log("No episodes in the replay buffer to train the H-network. Skipping training.")
            return None, None

        # create h_network sequences from the replay buffer
        input_sequences, target_sequences = self._create_h_network_sequences(_buffer)

        # chunk, pad, and mask the sequences
        # this will return tensor objects
        chunked_padded_inputs, chunked_padded_targets, mask = self._chunk_pad_mask_sequences(input_sequences, target_sequences, chunk_size=self.H_NETWORK_MAXSEQLEN, stride=64, padding_value=0.0)      # what will be the correct/better stride? either 1/8 max seq length, or full max seq length?

        # create dataset object and data loader for training
        dataset = TensorDataset(chunked_padded_inputs, chunked_padded_targets, mask)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.h_network.train()  # Set the H-network to training mode

        loss_list = []  # List to store loss values for each batch
        avg_loss = 0.0

        for i, batch in enumerate(dataloader):
            # print_log(f"Training H-network {i + 1}/{len(dataloader)} batches...", end='	')
            inputs, targets, mask = batch
            inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)
            self.h_network_training_optimizer.zero_grad()  # Zero the gradients

            mean, log_var = None, None
            lstm_1_h, lstm_2_h = None, None

            if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                mean, lstm_1_h, lstm_2_h = self.h_network(inputs)
            elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                mean, log_var, lstm_1_h, lstm_2_h = self.h_network(inputs)

            mean = mean.view(-1)
            targets = targets.view(-1)
            mask = mask.view(-1)

            mean_masked = torch.masked_select(mean, mask)
            targets_masked = torch.masked_select(targets, mask)

            if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                loss = self.h_network_training_criterion(mean_masked, targets_masked, var=torch.ones_like(mean_masked))
            elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                log_var_masked = torch.masked_select(log_var.view(-1), mask)
                loss = self.h_network_training_criterion(mean_masked, targets_masked, var=torch.exp(log_var_masked))

            loss.backward()  # Backward pass
            self.h_network_training_optimizer.step()  # Update weights

            if self.device == 'cuda':
                loss = loss.cpu()

            loss_list.append(loss.item())  # Store loss for this batch

        avg_loss = np.mean(loss_list)           # Calculate average loss over all batches
        std_loss = np.std(loss_list)             # Calculate standard deviation of loss over all batches
        self.train_loss_list.append({
            'mean_loss': float(avg_loss),
            'std_loss': float(std_loss),
            'num_batches': int(len(dataloader))
        })       # Append the average and std loss to the training loss list
        # print_log(f"Training H-network completed. Total batches: {len(dataloader)}; Average training loss: {avg_loss:.10f}")
        # print()

        # after training, reset the replay buffer
        self.replay_buffer.clear()

    def compute_f_signal(self, h_network_input, h_network_target):
        """
        Compute the f signal using the H-network.
        Args:
            h_network_input: torch.Tensor, input to the H-network
            h_network_target: torch.Tensor, target for the H-network
        Returns:
            f_signal: torch.Tensor, the computed f signal
            additional_info: dict, contains additional information about the last predictions
        """

        # no padding is needed, we directly reshape the input and target
        # and feed them to the H-network in one go
        # since LSTM should be able to handle variable length sequences (i.e. length that different from H_NETWORK_MAXSEQLEN)
        h_network_input, h_network_target = h_network_input.reshape(1, -1, 2), h_network_target.reshape(1, -1)
        
        # Convert to torch tensors
        h_network_input = torch.tensor(h_network_input, dtype=torch.float32, device=self.device)
        h_network_target = torch.tensor(h_network_target, dtype=torch.float32, device=self.device)

        self.h_network.eval()  # Set the H-network to evaluation mode

        signal_output = None

        with torch.no_grad():
           
            mean, log_var = None, None
            lstm_1_h, lstm_2_h = None, None

            if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                # H-Network that predicts only the mean
                mean, lstm_1_h, lstm_2_h = self.h_network(h_network_input, lstm_1_h, lstm_2_h)
            elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                # H-Network that predicts both mean and log variance
                mean, log_var, lstm_1_h, lstm_2_h = self.h_network(h_network_target, lstm_1_h, lstm_2_h)


            # flatten the mean and target
            mean = mean.view(-1)  # flatten the mean
            _target = h_network_target.view(-1)  # flatten the target


            if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                loss = self.h_network_criterion(mean, _target, var=torch.ones_like(mean))  # assuming unit variance

            elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                # For H-Network2, we need to compute the variance from log variance
                log_var = torch.exp(log_var)
                loss = self.h_network_criterion(mean, _target, var=log_var)

            # compute likelihood of each sample
            likelihood = torch.exp(-loss)

            # signal output is expected value of the negative loss
            signal_output = torch.mul(likelihood, -loss)

        # return the loss. This approximates \mathbb{E} \log(p(y_t_plus_1 | z_t, y_t)) -> the sum of signals approximates the Mutual Information (MI)
        signal_output = signal_output[-1]        # only consider the last sample.
        return signal_output.item(), \
                {'predicted_mean': float(mean[-1].item()), 
                 'predicted_log_var': float(log_var[-1].item()) if self.h_network_type.value == HNetworkType.H_NETWORK2.value else 0,   # for H-Network, we fixed the variance to 1.0 -> log_var is 0
                 'target': float(_target[-1].item())
                }