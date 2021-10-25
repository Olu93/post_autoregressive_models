import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modules.ConvolutionalBlock import ConvolutionalBlock


class ResidualBlockModule(torch.nn.Module):
    def __init__(self, in_emb_size, out_emb_size, kernel_size, dilation_size, stride=1, dilation=1):
        super(ResidualBlockModule, self).__init__()
        self.temporal_conv = ConvolutionalBlock(
            in_emb_size=in_emb_size,
            out_emb_size=out_emb_size,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv1x1 = nn.Conv1d(
            in_channels=in_emb_size,
            out_channels=out_emb_size,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
        )
        self.__padding = (kernel_size - 1) * dilation
        self.pad = nn.ZeroPad2d((self.__padding, 0))
        self.relu = nn.ReLU()

    def forward(self, input):
        # print("input")
        # print(input.shape)
        padded_input = self.pad(input)
        # print("padded")
        # print(padded_input.shape)
        temp_out = self.temporal_conv(padded_input)
        orig_out = self.conv1x1(input)
        # print("temp")
        # print(temp_out.shape)
        # print("orig")
        # print(orig_out.shape)
        residual = temp_out + orig_out
        return self.relu(residual)


if __name__ == '__main__':
    batch_size = 1
    n_channels = 3
    len_sequence = 10
    kernel_size = 2
    input = torch.from_numpy(
        np.arange(batch_size * n_channels * len_sequence).reshape((batch_size, n_channels, len_sequence))).float()
    # print(input.shape)
    module = ResidualBlockModule(in_emb_size=n_channels, out_emb_size=n_channels, kernel_size=kernel_size, dilation_size=1, dilation=1)
    out = module(input)
    print(out.shape)