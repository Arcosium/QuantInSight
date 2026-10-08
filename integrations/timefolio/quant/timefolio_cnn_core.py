"""Leakage-checked building blocks for the new long-only CNN study.

No broker imports, data discovery, downloads or training run on import.
Input panels must separately pass venue/actions/universe readiness checks.
ListNet uses its top-one cross entropy; topk_ce is a distinct research surrogate.
Reference: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def heatmap(ohlcv):
    """Encode one observed window as 1 x 8 x T; no data beyond its last bar.

    Columns are O,H,L,C,V. Price rows are log ratios to the first close;
    volume rows use only the supplied window. Corporate actions must already
    be reconciled as of the signal date. Missing bars are never interpolated.
    """
    a = np.asarray(ohlcv, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 5 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('A finite T x 5 observed window is required')
    o, h, l, c, v = a.T
    if (np.any(a[:, :4] <= 0) or np.any(v < 0) or v.sum() <= 0
            or np.any(l > np.minimum(o, c)) or np.any(h < np.maximum(o, c))):
        raise ValueError('Invalid OHLCV window')
    # Scales are fixed research constants, not fitted on future windows.
    prices = np.log(a[:, :4].T / c[0]) / .25
    volume = np.log((v + 1) / (np.median(v) + 1)) / 3
    intraday = np.log(c / o) / .1
    spread = np.log(h / l) / .1
    gap = np.zeros(len(c)); gap[1:] = np.log(o[1:] / c[:-1]) / .1
    return np.clip(np.vstack([prices, volume, intraday, spread, gap]), -1, 1).astype(np.float32)[None]


class RankCNN(nn.Module):
    """Small CNN; four time bins retain position within the observed chart."""
    def __init__(self, width=16, dropout=.1):
        super().__init__()
        if width < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid model configuration')
        self.features = nn.Sequential(
            nn.Conv2d(1, width, (3, 5), padding=(1, 2)), nn.LeakyReLU(.1),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.LeakyReLU(.1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 8, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] not in (8, 16, 22) or x.shape[-1] < 4:
            raise ValueError('N x 1 x H x T images with H in8,16,22 and T >= 4 required')
        z = self.features(x).mean(-2)
        n = z.shape[-1]
        z = torch.stack([z[..., i*n//4:(i+1)*n//4].mean(-1) for i in range(4)], -1)
        return self.head(z.flatten(1)).squeeze(1)


def date_loss(scores, returns, objective, *, top_k=10, temperature=.2):
    """One complete eligible date group per update; equal weight per date.

    The top-k positive-set CE includes every member tied at the cutoff. All
    other scores remain in its denominator, so it is not an exact top-k loss.
    Rank-based targets cannot tell the cash gate whether absolute returns win.
    """
    if (scores.ndim != 1 or scores.shape != returns.shape or len(scores) < 2
            or not scores.is_floating_point() or scores.device != returns.device
            or not torch.isfinite(scores).all() or not torch.isfinite(returns).all()):
        raise ValueError('One finite same-device date group with at least two stocks required')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature required')
    y = returns.detach().to(scores.dtype)
    if objective == 'mse':
        return F.mse_loss(scores, y)
    if objective == 'bce':
        return F.binary_cross_entropy_with_logits(scores, (y > 0).to(scores.dtype))
    if objective == 'listnet':
        # Stable midranks preserve ties; memory is linear in the universe size.
        _, inv, counts = torch.unique(y, sorted=True, return_inverse=True, return_counts=True)
        mid = counts.cumsum(0) - (counts + 1) / 2
        relevance = mid[inv].to(scores.dtype) / (len(y) - 1)
        probabilities = torch.softmax(relevance / temperature, dim=0)
    elif objective == 'topk_ce':
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(y):
            raise ValueError('top_k must be within the date group')
        positive = y >= torch.topk(y, top_k).values[-1]
        probabilities = positive.to(scores.dtype) / positive.sum()
    else:
        raise ValueError('Unknown objective')
    return -(probabilities * F.log_softmax(scores, dim=0)).sum()


def purged_masks(signal_index, label_end_index, valid, *, validation_start, test_start, test_end):
    """Require realized labels strictly before the next stage's signal date.

    For H sessions held after next-session execution, label_end is t+H+1,
    not t+H. Price windows may overlap; their unknown outcomes may not.
    """
    s, e, ok = np.asarray(signal_index), np.asarray(label_end_index), np.asarray(valid)
    if (s.ndim != 1 or s.shape != e.shape or s.shape != ok.shape or ok.dtype != bool
            or not np.issubdtype(s.dtype, np.integer) or not np.issubdtype(e.dtype, np.integer)
            or np.any(e <= s) or not validation_start < test_start < test_end):
        raise ValueError('Invalid sample boundaries or masks')
    train = ok & (s < validation_start) & (e < validation_start)
    validation = ok & (s >= validation_start) & (s < test_start) & (e < test_start)
    refit = ok & (s < test_start) & (e < test_start)
    # Outer inference membership must not depend on future label availability.
    prediction = (s >= test_start) & (s < test_end)
    return dict(train=train, validation=validation, refit=refit, prediction=prediction)


@dataclass
class CashGate:
    """Ridge expectation of basket net return, trained on past OOF baskets.

    y is the already cost-adjusted absolute basket return minus cash return.
    The gate requests exposure; contest turnover and holdings belong to the
    subsequent account simulator and may make full cash contest-ineligible.
    """
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    fit_cutoff: int
    fitted_rows: int

    @classmethod
    def fit(cls, x, net_excess, *, signal_dates, feature_available_dates,
            label_available_dates, stage1_train_label_end, stage1_selection_label_end,
            fit_cutoff, alpha=10., minimum_rows=60):
        x, y = np.asarray(x, dtype=float), np.asarray(net_excess, dtype=float)
        s, f, end, train, selection = map(np.asarray, [signal_dates, feature_available_dates,
            label_available_dates, stage1_train_label_end, stage1_selection_label_end])
        if (x.ndim != 2 or y.shape != (len(x),) or x.shape[1] < 1
                or any(a.shape != y.shape for a in [s, f, end, train, selection])
                or any(not np.issubdtype(a.dtype, np.integer) for a in [s, f, end, train, selection])
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(alpha) or alpha <= 0 or minimum_rows < 2):
            raise ValueError('Invalid gate arrays or configuration')
        if (len(np.unique(s)) != len(s) or np.any(f > s) or np.any(end <= s)
                or np.any(train >= s) or np.any(selection >= s)):
            raise ValueError('Gate needs unique, point-in-time, strictly OOF basket forecasts')
        use = (s < fit_cutoff) & (end < fit_cutoff)
        if use.sum() < minimum_rows:
            raise ValueError('Insufficient matured OOF basket dates for cash gate')
        mean, scale = x[use].mean(0), x[use].std(0)
        scale = np.maximum(scale, 1e-8)
        z = (x[use] - mean) / scale
        intercept = float(y[use].mean())
        beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y[use] - intercept))
        return cls(mean, scale, beta, intercept, int(fit_cutoff), int(use.sum()))

    def predict(self, x, *, signal_dates, feature_available_dates):
        x, dates, available = np.asarray(x, dtype=float), np.asarray(signal_dates), np.asarray(feature_available_dates)
        if (x.ndim != 2 or x.shape[1:] != self.mean.shape or not np.isfinite(x).all()
                or dates.shape != (len(x),) or available.shape != dates.shape
                or not np.issubdtype(dates.dtype, np.integer) or not np.issubdtype(available.dtype, np.integer)
                or np.any(dates < self.fit_cutoff) or np.any(available > dates)):
            raise ValueError('Gate prediction must use available features at or after fit cutoff')
        return ((x - self.mean) / self.scale) @ self.coefficient + self.intercept
