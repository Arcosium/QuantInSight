"""Small temporal-region heatmap encoders with same-date stock interaction.

This is a CNN adaptation motivated by stock-relation research, not a reproduction
of MASTER. Every attention batch is one signal date's full eligible feature set;
label availability selects supervised outputs, never the context constituents.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class StockRelationNet(nn.Module):
    def __init__(self, architecture: str, relation: str):
        super().__init__()
        if architecture not in ['cnn', 'mlp'] or relation not in ['self', 'mean', 'attention']:
            raise ValueError('Known architecture and stock-relation mode required')
        self.architecture = architecture; self.relation = relation
        if architecture == 'cnn':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(.1), nn.AdaptiveAvgPool2d((2, 5)))
            self.project = nn.Linear(64, 32)
        else:
            self.encoder = nn.Sequential(nn.Linear(96, 32), nn.LeakyReLU(.1), nn.Linear(32, 32), nn.LeakyReLU(.1))
        self.stock_norm = nn.LayerNorm(32)
        self.qkv = nn.Linear(32, 96, bias=False)
        self.stock_output = nn.Linear(32, 32, bias=False)
        self.feedforward = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.LeakyReLU(.1),
                                         nn.Dropout(.1), nn.Linear(64, 32))
        self.time_query = nn.Linear(32, 32, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 32, 65) or not 1 <= len(x) <= 512:
            raise ValueError('One date of1..512 normalized32x65 heatmaps required')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Finite normalized floating-point images required')
        if self.architecture == 'cnn':
            # Adaptive pooled regions overlap; these are not exact day partitions.
            encoded = self.encoder(x).permute(0, 3, 1, 2).reshape(len(x), 5, 64)
            encoded = self.project(encoded)
        else:
            chunks = x[:, 0].reshape(len(x), 32, 5, 13).permute(0, 2, 1, 3)
            summaries = torch.cat([chunks[..., -1], chunks.mean(-1), chunks.std(-1, correction=0)], dim=-1)
            encoded = self.encoder(summaries)
        regions = encoded.transpose(0, 1)  # region, stock, embedding
        q, k, v = self.qkv(self.stock_norm(regions)).chunk(3, dim=-1)
        if self.relation == 'self': mixed = v
        elif self.relation == 'mean': mixed = v.mean(1, keepdim=True).expand_as(v)
        else:
            weights = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(32.), dim=-1)
            mixed = weights @ v
        hidden = regions + self.stock_output(mixed)
        hidden = (hidden + self.feedforward(hidden)).transpose(0, 1)
        query = self.time_query(hidden[:, -1])
        time_weights = torch.softmax((hidden * query[:, None]).sum(-1) / np.sqrt(32.), dim=-1)
        pooled = (hidden * time_weights[..., None]).sum(1)
        return self.head(pooled).squeeze(-1)


def date_contexts(day_index, requested_mask):
    """Return full feature groups and the positions of supervised/requested rows."""
    days = np.asarray(day_index); requested = np.asarray(requested_mask)
    if days.ndim != 1 or not np.issubdtype(days.dtype, np.integer) or requested.dtype != bool or requested.shape != days.shape:
        raise ValueError('Integer date indices and equally sized boolean row mask required')
    groups = []
    for day in np.unique(days[requested]):
        context = np.flatnonzero(days == day)
        if len(context) > 512: raise ValueError('Unexpectedly large signal-date context')
        supervised = np.flatnonzero(requested[context])
        groups.append((context, supervised))
    return groups


def predict_by_date(model, x, day_index, indices):
    """Predict requested rows while preserving all as-of peers on each date."""
    ids = np.asarray(indices); days = np.asarray(day_index)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(x)) or days.shape != (len(x),)):
        raise ValueError('Unique in-range prediction indices and matching dates required')
    if x.dtype != torch.uint8: raise ValueError('Archived uint8 images required')
    requested = np.zeros(len(x), bool); requested[ids] = True
    groups = date_contexts(days, requested)
    values = np.full(len(x), np.nan, np.float32); model.eval()
    with torch.no_grad():
        for context, supervised in groups:
            scores = model(x[context].float().div(127.5).sub(1)).cpu().numpy()
            if scores.shape != (len(context),) or not np.isfinite(scores).all():
                raise AssertionError('Invalid stock-relation predictions')
            values[context[supervised]] = scores[supervised]
    return values[ids]
