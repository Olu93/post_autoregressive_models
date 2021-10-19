import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.TCN import ResidualTCN


class TextTCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2):
        super(TextTCNModel, self).__init__()
        assert num_channels[-1] == input_size, f"Out emb size must be the same as input emb size. Given: {num_channels[-1]}, {input_size}"
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = ResidualTCN(input_size, num_channels, kernel_size)
        self.decoder = nn.Linear(num_channels[-1], output_size)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.encoder(input)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()