import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torchinfo import summary

from modules.ResidualBlock import ResidualBlockModule


class TemporalConvolutionNetworkModel(nn.Module):
    def __init__(self, level_embedding_sizes, kernel_size=2):
        super(TemporalConvolutionNetworkModel, self).__init__()
        layers = []
        self.n_levels = len(level_embedding_sizes)
        for i in range(self.n_levels - 1):
            dilation = 1
            stride = 1 if i == 0 else 2
            in_emb_size = level_embedding_sizes[i]
            out_emb_size = level_embedding_sizes[i + 1]
            layers.append(self._add_tcn_module(kernel_size, dilation, stride, in_emb_size, out_emb_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        return ResidualBlockModule(
            in_emb_size=in_emb_size,
            out_emb_size=out_emb_size,
            kernel_size=kernel_size,
            dilation_size=stride,
            stride=stride,
            dilation=dilation,
        )


class SequentialTemporalConvolutionNetworkModel(nn.Module):
    def __init__(self, level_embedding_sizes, kernel_size=2):
        super(SequentialTemporalConvolutionNetworkModel, self).__init__()
        layers = []
        self.n_levels = len(level_embedding_sizes)
        for i in range(self.n_levels - 1):
            dilation = 2**(i)
            stride = 1
            in_emb_size = level_embedding_sizes[i]
            out_emb_size = level_embedding_sizes[i + 1]
            layers.append(self._add_tcn_module(kernel_size, dilation, stride, in_emb_size, out_emb_size))

        self.network = nn.Sequential(*layers)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        return ResidualBlockModule(
            in_emb_size=in_emb_size,
            out_emb_size=out_emb_size,
            kernel_size=kernel_size,
            dilation_size=stride,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    batch_size = 3
    n_channels = 5
    len_sequence = 16
    kernel_size = 2
    input = torch.from_numpy(
        np.arange(batch_size * n_channels * len_sequence).reshape((batch_size, n_channels, len_sequence))).float()
    # print(input.shape)
    # ss = (n_channels,)
    # module = ResidualTCN(level_embedding_sizes=ss, kernel_size=kernel_size)
    # out = module(input)
    # print(out.shape)
    # ss = (n_channels, 5)
    # module = ResidualTCN(level_embedding_sizes=ss, kernel_size=kernel_size)
    # out = module(input)
    # print(out.shape)
    # ss = (n_channels, 5, 5)
    # module = ResidualTCN(level_embedding_sizes=ss, kernel_size=kernel_size)
    # out = module(input)
    # print(out.shape)
    ss = (n_channels, 5, 5, 5, 5, 5)
    module = SequentialTemporalConvolutionNetworkModel(level_embedding_sizes=ss, kernel_size=kernel_size)
    out = module(input)
    print(summary(module, input_size=input.shape))