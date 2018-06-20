import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layer import DenseBlock, ResidualBlockUp


class Transformer(nn.Module):
    """ Refer to 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'.
        This module try to transfer image to target image.
    """

    def __init__(self, dim=32):
        super(Transformer, self).__init__()

        self.dim = dim
        # Encode
        self.conv1 = nn.Conv2d(3, self.dim, 9, padding=4)
        self.conv2 = nn.Conv2d(self.dim, self.dim * 2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.dim * 2, self.dim * 4, 4, stride=2, padding=1)
        self.conv_block1 = DenseBlock(self.dim * 4, self.dim * 4)
        self.conv_block2 = DenseBlock(self.dim * 4, self.dim * 4)
        self.conv_block3 = DenseBlock(self.dim * 4, self.dim * 4)
        self.conv_block4 = DenseBlock(self.dim * 4, self.dim * 4)
        self.conv_block5 = DenseBlock(self.dim * 4, self.dim * 4)
        # Decode
        self.dconv1 = nn.ConvTranspose2d(self.dim * 4, self.dim * 2, 4, stride=2, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(self.dim * 2, self.dim, 4, stride=2, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(self.dim * 4, 3, 9, stride=1, output_padding=4)

        self.bn1 = nn.BatchNorm2d(num_features=self.dim)
        self.bn2 = nn.BatchNorm2d(num_features=self.dim * 2)
        self.bn3 = nn.BatchNorm2d(num_features=self.dim * 4)
        self.bn4 = nn.BatchNorm2d(num_features=self.dim * 2)
        self.bn5 = nn.BatchNorm2d(num_features=self.dim * 4)

        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.bn1(F.elu(self.conv1(x)))
        h = self.bn2(F.elu(self.conv2(h)))
        h = self.bn3(F.elu(self.conv3(h)))
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.conv_block4(h)
        h = self.conv_block5(h)
        h = self.bn4(F.elu(self.dconv1(h)))
        h = self.bn5(F.elu(self.dconv2(h)))
        y = self.dconv3(h)

        return self.tanh(y)


class Generator(nn.Module):
    """ Transfer noise(128 dims) to target image. """

    def __init__(self, dim=32):
        super(Generator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4 * 4 * 8 * self.dim)
        self.conv_block1 = ResidualBlockUp(self.dim * 8, self.dim * 8)  # 8 * 8
        self.conv_block2 = ResidualBlockUp(self.dim * 8, self.dim * 4)  # 16 * 16
        self.conv_block3 = ResidualBlockUp(self.dim * 4, self.dim * 4)  # 32 * 32
        self.conv_block4 = ResidualBlockUp(self.dim * 4, self.dim * 2)  # 64 * 64
        self.conv_block5 = ResidualBlockUp(self.dim * 2, self.dim * 1)  # 128 * 128
        # self.conv_block6 = ResidualBlockUp(self.dim * 2, self.dim)  # 256 * 256
        self.conv1 = nn.Conv2d(self.dim, 3, 3, stride=1, padding=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.ln1(x.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        output = self.conv_block5(output)
        # output = self.conv_block6(output)
        output = self.conv1(output)

        return self.tanh(output)
