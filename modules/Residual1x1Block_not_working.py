import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from TemporalBlock import TemporalBlockModule
from CausalConvolution import CausalConv1d

class Residual1x1BlockModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(Residual1x1BlockModule, self).__init__()
        self.temporal_conv = TemporalBlockModule(in_channels, out_channels, kernel_size, stride=1)
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1, padding='valid')
        self.conv_1x1_2 = CausalConv1d(out_channels, out_channels, 1)
        
        self.__padding = min([(kernel_size - 1) * dilation, 1])
        self.pad = nn.ZeroPad2d((self.__padding, 0))

    def forward(self, input):
        padded_input = self.pad(input)
        # residual_out = self.conv_1x1_2(self.conv_1x1(padded_input))
        temporal_out = self.temporal_conv(padded_input)
        residual_out = self.conv_1x1(input)
        print(f"input: {input.shape}")
        print(f"padded: {padded_input.shape}")
        print(f"temporal: {temporal_out.shape}")
        print(f"residual: {residual_out.shape}")
        result = temporal_out + residual_out

        return result


if __name__ == '__main__':
    batch_size = 1
    n_channels = 1
    len_sequence = 10
    kernel_size = 2
    input = torch.from_numpy(
        np.arange(batch_size * n_channels * len_sequence).reshape((batch_size, n_channels, len_sequence))).float()
    print(input)
    module = Residual1x1BlockModule(in_channels=n_channels, out_channels=6, kernel_size=kernel_size, dilation=1)
    out = module(input)
    print(out.shape)