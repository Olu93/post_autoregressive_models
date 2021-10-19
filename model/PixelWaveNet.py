import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import optim

from modules.CausalConvolution import CausalConv1d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class PixelWaveNet(nn.Module):
    def __init__(self):
        super(PixelWaveNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            CausalConv1d(in_emb_size=1, out_emb_size=6, kernel_size=2, dilation=1),
            CausalConv1d(in_emb_size=6, out_emb_size=6, kernel_size=2, dilation=2),
            CausalConv1d(in_emb_size=6, out_emb_size=6, kernel_size=2, dilation=3),
            CausalConv1d(in_emb_size=6, out_emb_size=3, kernel_size=2, dilation=4),
            nn.Flatten(),
            nn.Linear(),
            # nn.ReLU(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        pixel_distribution = F.log_softmax(logits)
        return pixel_distribution

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)


if __name__ == '__main__':

    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    model = PixelWaveNet().to(device)
    print(model)
    network = PixelWaveNet()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)