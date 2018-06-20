import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layer import ResidualBlockDown


class Discriminator(nn.Module):

    def __init__(self, dim=32):
        super(Discriminator, self).__init__()

        self.dim = dim

        self.conv1 = nn.Conv2d(3, self.dim, 3, stride=1, padding=1)  # 256 * 256
        self.conv_block1 = ResidualBlockDown(self.dim * 1, self.dim * 2)  # 128 * 128
        self.conv_block2 = ResidualBlockDown(self.dim * 2, self.dim * 2)  # 64 * 64
        self.conv_block3 = ResidualBlockDown(self.dim * 2, self.dim * 4)  # 32 * 32
        self.conv_block4 = ResidualBlockDown(self.dim * 4, self.dim * 4)  # 16 * 16
        self.conv_block5 = ResidualBlockDown(self.dim * 4, self.dim * 8)  # 8 * 8
        # self.conv_block6 = ResidualBlockDown(self.dim * 8, self.dim * 8)  # 4 * 4
        self.conv2 = nn.Conv2d(self.dim * 8, self.dim * 8, 3, stride=1, padding=0)

    def forward(self, x):
        # Size of Image x is 256 * 256.
        b, c, w, h = x.shape
        output = self.conv1(x)
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        output = self.conv_block5(output)
        # output = self.conv_block6(output)
        output = self.conv2(output)
        output = output.view(b, -1)

        return output.mean(dim=1)
