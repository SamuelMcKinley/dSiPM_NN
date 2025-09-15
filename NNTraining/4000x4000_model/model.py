import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyRegressionCNN(nn.Module):
    """
    Sturdy CNN for [B, C, H, W] inputs producing a scalar regression output.
    C is typically the number of time slices (channels) from your photon tensor.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels is None or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x)  # [B, 1]
        return x.squeeze(-1)  # [B]

