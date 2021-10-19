import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from modules.CausalConvolution import CausalConv1d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class TextWaveNet(nn.Module):
    def __init__(self):
        super(TextWaveNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            CausalConv1d(in_emb_size=1, out_emb_size=6, kernel_size=2, dilation=1),
            CausalConv1d(in_emb_size=6, out_emb_size=6, kernel_size=2, dilation=2),
            CausalConv1d(in_emb_size=6, out_emb_size=6, kernel_size=2, dilation=3),
            CausalConv1d(in_emb_size=6, out_emb_size=3, kernel_size=2, dilation=4),
            nn.Flatten(),
            nn.Linear(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        pixel_distribution = F.log_softmax(logits)
        return pixel_distribution


if __name__ == '__main__':
    model = TextWaveNet().to(device)
    print(model)
