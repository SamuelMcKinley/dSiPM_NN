#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k=3, s=1, p=1, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class EnergyRegressionCNN(nn.Module):
    """
    Simple, robust CNN for regression on (C,H,W) tensors.
    Uses global average pooling before a small MLP head.
    """

    def __init__(self, in_channels: int, hidden: int = 128, bn: bool = True, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(in_channels, 32, k=3, s=1, p=1, bn=bn),
            conv_block(32, 64, k=3, s=2, p=1, bn=bn),   # downsample
            conv_block(64, 128, k=3, s=2, p=1, bn=bn),  # downsample
            conv_block(128, 128, k=3, s=1, p=1, bn=bn),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.gap(x)
        x = self.head(x)
        return x.squeeze(-1)  # (N,)