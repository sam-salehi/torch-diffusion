import torch.nn as nn
from utils import T


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x, t):
        t_embed = t[:, None, None, None].float() / T
        return self.net(x + t_embed)
