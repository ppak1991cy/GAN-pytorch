import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):

    def __init__(self, d_in, d_out):
        super(DenseBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(d_in)
        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class ResidualBlockDown(nn.Module):
    """ Residual block used to down sampling """

    def __init__(self, d_in, d_out):
        super(ResidualBlockDown, self).__init__()

        self.conv = nn.Conv2d(d_in, d_out, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(d_in)
        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(d_out)
        self.conv2 = nn.Conv2d(d_out, d_out, stride=2, kernel_size=3, padding=1)

    def forward(self, x):
        x_down = self.conv(F.avg_pool2d(x, 2))
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + x_down
        return out


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        input = torch.cat((input, input, input, input), 1)
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, output_height,
                                                                                       output_width, output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class ResidualBlockUp(nn.Module):
    """ Residual block used to up sampling """

    def __init__(self, d_in, d_out):
        super(ResidualBlockUp, self).__init__()

        self.conv = nn.Conv2d(d_in, d_out, kernel_size=1, padding=0)
        self.up = DepthToSpace(2)
        self.bn1 = nn.BatchNorm2d(d_in)
        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(d_out)
        self.conv2 = nn.ConvTranspose2d(d_out, d_out, stride=2, kernel_size=4, padding=1)

    def forward(self, x):
        x_up = self.conv(x)
        x_up = self.up(x_up)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + x_up
        return out
