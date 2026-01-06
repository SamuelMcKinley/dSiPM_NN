import torch.nn as nn

class PhotonMLP(nn.Module):
    def __init__(self, hidden=(32, 32), dropout=0.0):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B,1)