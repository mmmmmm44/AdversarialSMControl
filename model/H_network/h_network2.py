# The H-network from the paper Privacy-Cost Management in Smart Meters with Mutual-Information-Based Reinforcement Learning.
# It predicts both mean and log variance of the output distribution.

import torch
import torch.nn as nn
import torch.nn.functional as F

class HNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HNetwork, self).__init__()
        self.LSTM_1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ac1 = nn.Tanh()
        self.LSTM_2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.ac2 = nn.Tanh()
        self.fc_mu = nn.Linear(hidden_dim * 2, output_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, LSTM_1_h = None, LSTM_2_h = None):

        if LSTM_1_h is None:
            LSTM_1_h = self._get_hidden(x, self.LSTM_1.hidden_size)
        
        if LSTM_2_h is None:
            LSTM_2_h = self._get_hidden(x, self.LSTM_2.hidden_size)

        x, lstm_1_h = self.LSTM_1(x, LSTM_1_h)
        x = self.ac1(x)
        x, lstm_2_h = self.LSTM_2(x, LSTM_2_h)
        x = self.ac2(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        mu = mu.squeeze(-1)  # flatten the output to shape [batch, seq_len]
        log_var = log_var.squeeze(-1)  # flatten the output to shape [batch, seq_len]
        return mu, log_var, lstm_1_h, lstm_2_h
    
    def _get_hidden(self, x, hidden_size):
        hidden = (
            torch.zeros(2, x.shape[0], hidden_size, device=x.device),
            torch.zeros(2, x.shape[0], hidden_size, device=x.device)
        )
        return hidden