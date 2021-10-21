import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys 

if __name__ != '__main__':
    from modules.TemporalBlock import TemporalBlockModule
else:
    from TemporalBlock import TemporalBlockModule



class PureTCN(nn.Module):
    def __init__(self, level_embedding_sizes, kernel_size=2, stride=1):
        super(PureTCN, self).__init__()
        layers = []
        self.n_levels = len(level_embedding_sizes)

        for i in range(self.n_levels - 1):
            dilation = 2**(i)
            in_emb_size = level_embedding_sizes[i]
            out_emb_size = level_embedding_sizes[i + 1]
            layers.append(self._add_tcn_module(kernel_size, dilation, stride, in_emb_size, out_emb_size))

        self.network = nn.Sequential(*layers)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        # print("params ", in_emb_size, out_emb_size, kernel_size, stride, dilation)
        return TemporalBlockModule(in_emb_size, out_emb_size, kernel_size, stride, dilation)

    def forward(self, x):
        return self.network(x)


class ResidualTCN(PureTCN):
    def __init__(self, level_embedding_sizes, kernel_size=2, stride=1):
        super(ResidualTCN, self).__init__(level_embedding_sizes, kernel_size, stride)

    def forward(self, x):
        return self.network(x)

    def _add_tcn_module(self, kernel_size, dilation, stride, in_emb_size, out_emb_size):
        return TemporalBlockModule(in_emb_size, out_emb_size, kernel_size, stride, dilation)


if __name__ == '__main__':
    batch_size = 3
    n_channels = 5
    len_sequence = 30
    kernel_size = 2
    input = torch.from_numpy(
        np.arange(batch_size * n_channels * len_sequence).reshape((batch_size, n_channels, len_sequence))).float()
    print(input.shape)
    ss = (n_channels, 6, 7, 6, 10, 10)
    module = ResidualTCN(level_embedding_sizes=ss, kernel_size=kernel_size)
    out = module(input)
    print(out.shape)