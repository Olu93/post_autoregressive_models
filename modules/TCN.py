import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ResidualBlock import ResidualBlockModule
from TemporalBlock import TemporalBlockModule


class PureTCN(nn.Module):
    def __init__(self, num_inputs, level_embedding_sizes, kernel_size=2, stride=1):
        super(PureTCN, self).__init__()
        layers = []
        all_level_embedding_sizes = [num_inputs] + list(level_embedding_sizes)
        n_levels = len(all_level_embedding_sizes)

        for i in range(n_levels - 1):
            dilation = 2**(i)
            in_emb_size = all_level_embedding_sizes[i]
            out_emb_size = all_level_embedding_sizes[i + 1]
            layers.append(self._add_tcn_module(kernel_size, dilation, stride, in_emb_size, out_emb_size))

        self.network = nn.Sequential(*layers)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        return TemporalBlockModule(in_emb_size, out_emb_size, kernel_size, stride, dilation)

    def forward(self, x):
        return self.network(x)


class ResidualTCN(PureTCN):
    def __init__(self, num_inputs, level_embedding_sizes, kernel_size=2, stride=1):
        super(ResidualTCN, self).__init__(num_inputs, level_embedding_sizes, kernel_size, stride)

    def forward(self, x):
        return self.network(x)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        return TemporalBlockModule(in_emb_size, out_emb_size, kernel_size, stride, dilation)


if __name__ == '__main__':
    batch_size = 3
    n_channels = 5
    len_sequence = 10
    kernel_size = 2
    input = torch.from_numpy(
        np.arange(batch_size * n_channels * len_sequence).reshape((batch_size, n_channels, len_sequence))).float()
    print(input.shape)
    module = PureTCN(num_inputs=batch_size, level_embedding_sizes=(6, 6, 8), kernel_size=kernel_size)
    out = module(input)
    print(out.shape)