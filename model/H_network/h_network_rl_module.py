# The module which will be integrated into the RL training loop to realize loop-training of H-network, along with the RL agent.

from collections import deque

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np

from rl_env.hrl_env_episode import SmartMeterEpisode

from h_network_arch import HNetworkType
from h_network import HNetwork as HNetwork
from h_network2 import HNetwork as HNetwork2

class HNetworkRLModule:
    """
    A module to handle the H-network training and inference in the context of reinforcement learning.
    This module is designed to be integrated into a reinforcement learning training loop.
    """

    def __init__(self, h_network_type: HNetworkType, device):
        self.device = device                    # Device to run the H-network on (e.g., 'cpu' or 'cuda')
        self.h_network_type = h_network_type

        self.h_network = None                   # Placeholder for H-network, to be loaded or trained
        self.H_NETWORK_MAXSEQLEN = 512          # Maximum sequence length for H-network inference buffer
        


        self.replay_buffer = deque()  # Replay buffer for H-network training, stores SmartMeterEpisode objects

        # RL environment related variables
        # self.h_network_stdscaler = h_network_stdscaler  # Placeholder for H-network standard scaler, to be loaded or trained
        self.h_network_criterion = nn.GaussianNLLLoss(reduction='none')  # loss function as the privacy signal.
        # self.h_network_inference_buffer = deque()  # Buffer for H-network inference, to store recent pair of input (z_t, y_t) and desired target (y_{t+1})


    def initialize_h_network(self):
        """
        Initialize the H-network. This replaces any existing H-network with a new one.
        """

        # don't know why the enum comparison does not work here, so we use value comparison right now...
        if self.h_network_type.value == HNetworkType.H_NETWORK.value:
            return HNetwork(input_dim=2, hidden_dim=44, output_dim=1)
        elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
            return HNetwork2(input_dim=2, hidden_dim=44, output_dim=1)
        else:
            raise ValueError(f"Unsupported H-network type: {self.h_network_type}. Supported types are: {list(HNetworkType)}")
        
    def initialize_h_network_training(self):
        """
        Initialize the H-network training components, including the loss function and optimizer.
        This method should be called after the H-network is set.
        """
        if self.h_network is None:
            raise ValueError("H-network is not set. Please set the H-network before initializing training components.")

        self.h_network_training_criterion = nn.GaussianNLLLoss()
        self.h_network_training_optimizer = torch.optim.Adam(self.h_network.parameters(), lr=1e-3)

        self.train_loss_list = []

        self.h_network.to(self.device)  # Move the H-network to the specified device
        
    def set_h_network(self, h_network):
        """
        Set the H-network to be used in the RL training loop.
        """
        self.h_network = h_network

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
            # check whether the episode has non-zero grid load ?
            

            episode.df['grid_load_std'] = episode.standard_scalar.transform(episode.df[['grid_load']].values).flatten()

            _buffer.append(episode.df[['grid_load_std', 'aggregate_std']].values)  # Store the standardized grid load and aggregate load

        if len(_buffer) == 0:
            # print_log("No episodes in the replay buffer to train the H-network. Skipping training.")
            return None, None

        # create h_network sequences from the replay buffer
        input_sequences, target_sequences = self._create_h_network_sequences(_buffer)

        # chunk, pad, and mask the sequences
        padded_inputs, padded_targets, mask = self._chunk_pad_mask_sequences(input_sequences, target_sequences, chunk_size=self.H_NETWORK_MAXSEQLEN, stride=64, padding_value=0.0)      # what will be the correct/better stride? either 1/8 max seq length, or full max seq length?

        # create dataset object and data loader for training
        dataset = TensorDataset(padded_inputs, padded_targets, mask)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        loss_list = []  # List to store loss values for each batch
        avg_loss = 0.0

        for i, batch in enumerate(dataloader):
            # print_log(f"Training H-network {i + 1}/{len(dataloader)} batches...", end='\t')
            inputs, targets, mask = batch
            inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)
            self.h_network_training_optimizer.zero_grad()  # Zero the gradients

            mean, log_var = None, None

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

            avg_loss += loss.item()
            loss_list.append(loss.item())  # Store loss for this batch

        avg_loss /= len(dataloader)
        self.train_loss_list.append(avg_loss)
        # print_log(f"Training H-network completed. Total batches: {len(dataloader)}; Average training loss: {avg_loss:.10f}")
        # print()

        # after training, reset the replay buffer
        self.replay_buffer.clear()

        return avg_loss, loss_list
    

    def compute_f_signal(self, h_network_input, h_network_target):
        """
        Compute the f signal using the H-network.
        Args:
            h_network_input: torch.Tensor, input to the H-network
            h_network_target: torch.Tensor, target for the H-network
        Returns:
            f_signal: torch.Tensor, the computed f signal
        """
        if len(h_network_input) > self.H_NETWORK_MAXSEQLEN:
            h_network_input, h_network_target, h_network_mask = self._chunk_pad_mask_sequences(
                h_network_input.reshape(1, -1, 2),
                h_network_target.reshape(1, -1),
                chunk_size=self.H_NETWORK_MAXSEQLEN,
                stride=self.H_NETWORK_MAXSEQLEN,
            )
        else:
            # no padding is needed
            h_network_input, h_network_target, h_network_mask = h_network_input.reshape(1, -1, 2), h_network_target.reshape(1, -1), np.ones(len(h_network_input), dtype=bool).reshape(1, -1)
            
            # Convert to torch tensors
            h_network_input = torch.tensor(h_network_input, dtype=torch.float32)
            h_network_target = torch.tensor(h_network_target, dtype=torch.float32)
            h_network_mask = torch.tensor(h_network_mask, dtype=torch.bool)

        dataloader = DataLoader(
            TensorDataset(h_network_input, h_network_target, h_network_mask),
            batch_size=64,
            shuffle=False
        )

        signal_output = None

        with torch.no_grad():
            lstm_1_h, lstm_2_h = None, None

            for i, batch in enumerate(dataloader):
                _input, _target, _mask = batch
                _input, _target, _mask = _input.to(self.device), _target.to(self.device), _mask.to(self.device)

                mean, log_var = None, None

                if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                    # H-Network that predicts only the mean
                    mean, lstm_1_h, lstm_2_h = self.h_network(_input, lstm_1_h, lstm_2_h)
                elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                    # H-Network that predicts both mean and log variance
                    mean, log_var, lstm_1_h, lstm_2_h = self.h_network(_input, lstm_1_h, lstm_2_h)

                if i != len(dataloader) - 1:
                    # If not the last batch, we do not compute the loss yet
                    continue

                # Apply the mask to the mean and target
                mean = mean.view(-1)  # flatten the mean
                _target = _target.view(-1)  # flatten the target
                _mask = _mask.view(-1)  # flatten the mask

                mean_masked = torch.masked_select(mean, _mask)  # apply the mask to the mean
                _target_masked = torch.masked_select(_target, _mask)  # apply the mask to the target

                if self.h_network_type.value == HNetworkType.H_NETWORK.value:
                    loss = self.h_network_criterion(mean_masked, _target_masked, var=torch.ones_like(mean_masked))  # assuming unit variance

                elif self.h_network_type.value == HNetworkType.H_NETWORK2.value:
                    # For H-Network2, we need to compute the variance from log variance
                    log_var_masked = torch.masked_select(log_var.view(-1), _mask)                                     # apply the mask to the log variance
                    loss = self.h_network_criterion(mean_masked, _target_masked, var=torch.exp(log_var_masked))    # compute the loss with the masked mean and target

                # compute likelihood of each sample
                likelihood = torch.exp(-loss)

                # signal output is expected value of the negative loss
                signal_output = torch.mul(likelihood, -loss)

        # return the loss. This approximates \mathbb{E} \log(p(y_t_plus_1 | z_t, y_t)) -> the sum of signals approximates the Mutual Information (MI)
        signal_output = signal_output[-1]        # only consider the last sample.
        return signal_output.item()
    
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

    def push_to_replay_buffer(self, episode: SmartMeterEpisode):
        """
        Push a completed episode to the replay buffer.
        Args:
            episode (SmartMeterEpisode): The completed episode to push.
        """
        self.replay_buffer.append(episode)
