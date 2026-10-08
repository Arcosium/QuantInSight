"""Top-k NDCG-weighted pairwise surrogate for a single signal date.

Reference: Burges, MSR-TR-2010-82, sections3 and4.1, equations5 and6:
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

This research adaptation maps within-date return ranks continuously to relevance
grades0..4. Pair weights are detached absolute NDCG@k swap differences, normalised
to equal total date weight. It is not a portfolio-return objective or evidence of
a financial edge. Stable security keys resolve tied model scores.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def ndcg_pair_weights(scores, target, *, top_k=12, security_keys=None):
    """Return directed winner/loser swap weights, without a gradient path."""
    if scores.ndim != 1 or target.shape != scores.shape or not 1 <= len(scores) <= 512:
        raise ValueError('One matching signal-date vector of1..512 samples required')
    if not scores.is_floating_point() or not torch.isfinite(scores).all() or not torch.isfinite(target).all():
        raise ValueError('Finite floating scores and finite targets required')
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('Positive integer top_k required')
    if scores.device != target.device:
        raise ValueError('Scores and targets must share a device')
    n = len(scores)
    if security_keys is None: security_keys = torch.arange(n, device=scores.device)
    if security_keys.shape != scores.shape or security_keys.device != scores.device or not torch.isfinite(security_keys).all():
        raise ValueError('Finite matching security keys on the score device required')
    if len(torch.unique(security_keys)) != n: raise ValueError('Security keys must be unique within the date')
    with torch.no_grad():
        if n == 1: return torch.zeros((1, 1), dtype=scores.dtype, device=scores.device)
        y = target.detach().to(torch.float64)
        winning = y[:, None] > y[None, :]
        equal = y[:, None] == y[None, :]
        midrank = winning.sum(1).to(torch.float64) + (equal.sum(1) - 1) / 2
        gain = torch.exp2(4 * midrank / (n - 1)) - 1
        keys = torch.argsort(security_keys, stable=True)
        order = keys[torch.argsort(scores.detach()[keys], descending=True, stable=True)]
        positions = torch.empty(n, dtype=torch.float64, device=scores.device)
        positions[order] = torch.arange(1, n + 1, dtype=torch.float64, device=scores.device)
        discount = torch.where(positions <= top_k, 1 / torch.log2(positions + 1), 0.)
        ideal_position = torch.arange(1, min(top_k, n) + 1, dtype=torch.float64, device=scores.device)
        ideal = (torch.sort(gain, descending=True).values[:top_k] / torch.log2(ideal_position + 1)).sum()
        if ideal <= 0: return torch.zeros((n, n), dtype=scores.dtype, device=scores.device)
        weights = (gain[:, None] - gain[None, :]).abs() * (discount[:, None] - discount[None, :]).abs() / ideal
        return torch.where(winning, weights, 0.).to(dtype=scores.dtype)


def top_rank_loss(scores, target, *, top_k=12, security_keys=None):
    """Normalised, locally weighted RankNet loss; no derivative through sorting."""
    weights = ndcg_pair_weights(scores, target, top_k=top_k, security_keys=security_keys)
    total = weights.sum()
    if total <= 0: return scores.sum() * 0
    margin = scores[:, None] - scores[None, :]
    return (F.softplus(-margin) * weights).sum() / total
