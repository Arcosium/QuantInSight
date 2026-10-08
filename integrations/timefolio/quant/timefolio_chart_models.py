"""Small image CNN with deterministic pooling and no pretrained data."""
import torch
from torch import nn
from quant.timefolio_chart_lab import SHAPE
from quant.timefolio_heatmap_lab_models import FixedPool


class ChartNet(nn.Module):
    def __init__(self, cfg):
        super().__init__(); w = cfg['width']
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
            nn.Conv2d(w, 2*w, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
            nn.Conv2d(2*w, 4*w, 3, padding=1), nn.LeakyReLU(.1), FixedPool(),
            nn.Flatten(), nn.Dropout(cfg['dropout']), nn.Linear(32*w, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, *SHAPE):
            raise ValueError('Registered chart geometry required')
        return self.net(x).squeeze(1)
