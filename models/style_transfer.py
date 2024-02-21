import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        return x1 + x


class StyleTransfer(nn.Module):
    """
    input_size: 3 * 256 * 256
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, 1, 4)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.RB1 = ResidualBlock(128)
        self.RB2 = ResidualBlock(128)
        self.RB3 = ResidualBlock(128)
        self.RB4 = ResidualBlock(128)
        self.RB5 = ResidualBlock(128)
        self.convT1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.convT2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.convT3 = nn.Conv2d(32, 3, 9, 1, 4)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.RB5(self.RB4(self.RB3(self.RB2(self.RB1(x)))))
        x = F.relu(self.bn4(self.convT1(x)))
        x = F.relu(self.bn5(self.convT2(x)))
        x = self.convT3(x)
        return x


