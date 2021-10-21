import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_emb_size, out_emb_size, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super(CausalConv1d, self).__init__(in_emb_size,
                                           out_emb_size,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)

        self.__padding = (kernel_size - 1) * dilation
        self.pad = nn.ZeroPad2d((self.__padding, 0))

    def forward(self, input):
        padded_input = self.pad(input)
        return super(CausalConv1d, self).forward(padded_input)


if __name__ == '__main__':
    batch_size = 1
    embedding_size = 11
    len_sequence = 5
    kernel_size = 2
    dilation = 1
    __padding = min([(kernel_size - 1) * dilation, 1])
    input = torch.from_numpy(
        np.arange(batch_size * embedding_size * len_sequence).reshape((batch_size, embedding_size, len_sequence))).float()
    print(f"inputs : {input.shape}")
    print(f"pad_in : {F.pad(input, (__padding, 0)).shape}")
    CaConv1d = CausalConv1d(in_emb_size=embedding_size, out_emb_size=embedding_size, kernel_size=kernel_size, dilation=dilation)
    out = CaConv1d(input)
    print(f"pad_out: {out.shape}")