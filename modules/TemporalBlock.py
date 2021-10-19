import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CausalConvolution import CausalConv1d


class TemporalBlockModule(nn.Module):
    def __init__(self, in_emb_size, out_emb_size, kernel_size, stride=1, dilation=1):
        super(TemporalBlockModule, self).__init__()
        self.in_emb_size = in_emb_size # Muss channel sein, ist aber anzahl
        self.out_emb_size = out_emb_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.run_block = nn.Sequential(*[
            nn.Conv1d(in_emb_size, out_emb_size, kernel_size, stride=stride, dilation=dilation, bias=True),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            # nn.Conv1d(out_emb_size, out_emb_size, kernel_size, stride=stride, dilation=dilation, bias=True),
            # nn.LazyBatchNorm1d(),
        ])

    def forward(self, input):
        print("=============")
        print("Input shape ",input.shape)
        print("In and Out ", self.in_emb_size, self.out_emb_size)
        print("kernel, stride, dilation ", self.kernel_size, self.stride, self.dilation)
        result = self.run_block(input)
        print("Output shape ",result.shape)
        print("=============")
        return result


if __name__ == '__main__':
    batch_size = 1
    embedding_size = 11
    len_sequence = 10
    kernel_size = 2
    dilation = 1
    __padding = min([(kernel_size - 1) * dilation, 1])
    input = torch.from_numpy(
        np.arange(batch_size * embedding_size * len_sequence).reshape((batch_size, embedding_size, len_sequence))).float()
    print(f"inputs : {input.shape}")
    print(f"pad_in : {F.pad(input, (__padding, 0)).shape}")
    module = TemporalBlockModule(in_emb_size=embedding_size, out_emb_size=embedding_size, kernel_size=kernel_size, dilation=1)
    out = module(F.pad(input, (1, 0)))
    print(out.shape)