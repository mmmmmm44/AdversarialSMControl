# The H-network from the paper Privacy-Cost Management in Smart Meters with Mutual-Information-Based Reinforcement Learning.

import torch
import torch.nn as nn
import torch.nn.functional as F

class HNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HNetwork, self).__init__()
        self.LSTM_1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ac1 = nn.Tanh()
        self.LSTM_2 = nn.LSTM(hidden_dim * 2, output_dim, batch_first=True, bidirectional=True)
        self.ac2 = nn.Tanh()
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.LSTM_1(x)
        x = self.ac1(x)
        x, _ = self.LSTM_2(x)
        x = self.ac2(x)
        x = self.fc(x)
        x = x.squeeze(-1)  # flatten the output to shape [batch, seq_len]
        return x