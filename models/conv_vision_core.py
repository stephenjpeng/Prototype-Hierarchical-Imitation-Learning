import torch
from torch import nn


class ConvVisionCore(nn.Module):
    def __init__(self, args):
        super(ConvVisionCore, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, args['hidden_size'], kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def reset(self):
        return

    def forward(self, x):
        x = x.transpose(1, 3) * 1./ 255.
        return self.conv(x).transpose(1, 3)
