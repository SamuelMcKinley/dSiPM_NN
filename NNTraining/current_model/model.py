#!/usr/bin/env python3
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, k=3, s=1, p=1, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class EnergyRegressionCNN(nn.Module):
    """
    CNN on normalized hit-map tensor x plus scalar lnN.
    Forward expects:
      x   : (N,C,H,W)
      lnN : (N,1) or (N,) or (N,)

    Output:
      y_norm_pred : (N,)  (normalized target space)
    """

    def __init__(self, in_channels: int, hidden: int = 128, bn: bool = True, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(in_channels, 32,  k=3, s=1, p=1, bn=bn),
            conv_block(32,          64,  k=3, s=2, p=1, bn=bn),
            conv_block(64,          128, k=3, s=2, p=1, bn=bn),
            conv_block(128,         128, k=3, s=1, p=1, bn=bn),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # +1 for lnN feature
        self.head = nn.Sequential(
            nn.Flatten(),                 # (N,128,1,1) -> (N,128)
            nn.Linear(128 + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lnN: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)
        f = self.gap(f).flatten(1)  # (N,128)

        # lnN -> (N,1)
        if lnN.ndim == 1:
            lnN = lnN.unsqueeze(1)
        elif lnN.ndim == 2 and lnN.shape[1] != 1:
            lnN = lnN[:, :1]
        lnN = lnN.to(device=f.device, dtype=f.dtype)

        h = torch.cat([f, lnN], dim=1)    # (N,129)
        out = self.head(h).squeeze(-1)    # (N,)
        return out