"""Rank feasible target portfolios instead of individual stocks.

Research adaptation of Mandi et al., ICML2022, sections4.2/4.5:
https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf
The paper motivates ranking feasible solutions. Here a per-date greedy pool
and exposure-normalised all-pair logistic loss replace its oracle pool and margin loss.
This is a fresh-account target surrogate, not a trade-ledger objective.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from quant.timefolio_heatmap_replay import targets


def feasible_pool(predicted, target, eligible, sector, sector_caps, market_cap, codes,
                  stock_caps, *, top_n=12, weight=.05, gross=.8, random_draws=16, seed=317):
    """Detached feasible weights from two observed orders and fixed random orders.

    Target labels may be passed only for already observed, purged training dates.
    No optimality claim is made for the greedy target ordering or the pool.
    Random scores are assigned in ticker order, preserving row permutation.
    """
    predicted, target = np.asarray(predicted), np.asarray(target)
    n = len(predicted)
    vectors = [target, eligible, sector, sector_caps, market_cap, codes, stock_caps]
    if predicted.ndim != 1 or not 1 <= n <= 512 or any(np.asarray(v).shape != (n,) for v in vectors):
        raise ValueError('Matching per-date vectors of1..512 securities required')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all(): raise ValueError('Finite scores and labels required')
    sector, caps, mc, sc = map(np.asarray, [sector, sector_caps, market_cap, stock_caps])
    names = np.asarray(codes).astype(str); admitted = np.asarray(eligible, bool)
    if len(set(names)) != n: raise ValueError('Unique security codes required')
    if not all(np.isfinite(v).all() for v in [sector, caps, mc, sc]): raise ValueError('Finite metadata required')
    if np.any(sector != sector.astype(np.int64)): raise ValueError('Integer sector labels required')
    if np.any((caps < 0) | (caps > 1)) or np.any((sc <= 0) | (sc > 1)) or np.any(mc < 0):
        raise ValueError('Invalid capitalisation or weight limits')
    if type(top_n) is not int or top_n < 1 or type(random_draws) is not int or not 0 <= random_draws <= 62:
        raise ValueError('Positive top_n and0..62 random orders required')
    if not np.isfinite(weight) or not 0 < weight <= 1 or not np.isfinite(gross) or not 0 < gross <= 1:
        raise ValueError('Positive finite stock and gross targets required')
    for sec in np.unique(sector):
        if np.ptp(caps[sector == sec]) > 1e-12: raise ValueError('Common limit required within each sector')
    canonical = np.argsort(names, kind='stable'); rng = np.random.default_rng(seed)
    orders = [predicted, target]
    for _ in range(random_draws):
        score = np.empty(n); score[canonical] = rng.standard_normal(n); orders.append(score)
    pool, seen = [], set()
    for score in orders:
        weights = targets(score, admitted, sector, caps, mc, names, top_n=top_n, weight=weight, gross=gross)
        weights = np.minimum(weights, sc)
        # Floating residual headroom may differ by machine precision across orders.
        signature = np.round(weights, 12).tobytes()
        if signature in seen: continue
        seen.add(signature); pool.append(weights)
    return np.stack(pool)


def feasible_pair_loss(scores, target, pool):
    """Mean RankNet loss over feasible portfolios ordered by observed utility.

    One signal date only. Frozen pool weights map stock outputs to portfolio
    utility per unit of invested weight. The raw feasible weights remain intact;
    only the utility calculation divides by each positive gross. This makes
    score/label translation irrelevant, matching a rank-only deployment rule.
    Empty portfolios are omitted; this is not a cash gate or total-NAV return
    objective. No derivatives flow into pool or labels.
    """
    if scores.ndim != 1 or target.shape != scores.shape or not 1 <= len(scores) <= 512:
        raise ValueError('Matching score and target vectors required')
    if pool.ndim != 2 or pool.shape[1] != len(scores) or not 1 <= len(pool) <= 64:
        raise ValueError('A pool of1..64 matching feasible portfolios required')
    if not scores.is_floating_point() or not target.is_floating_point() or not pool.is_floating_point():
        raise ValueError('Floating tensors required')
    if scores.device != target.device or scores.device != pool.device: raise ValueError('Tensors must share a device')
    if not all(torch.isfinite(v).all() for v in [scores, target, pool]) or (pool < 0).any():
        raise ValueError('Finite scores/labels and nonnegative finite weights required')
    if (pool.sum(1) > 1 + 1e-6).any(): raise ValueError('Portfolio gross exceeds one')
    weights = pool.detach().to(torch.float64)
    gross = weights.sum(1); positive = gross > 0
    if positive.sum() < 2: return scores.sum() * 0
    weights = weights[positive] / gross[positive, None]
    utility = weights @ target.detach().to(torch.float64)
    # Dot-product roundoff must not invent an ordering of equal utilities.
    tolerance = 1024 * torch.finfo(torch.float64).eps * torch.maximum(target.detach().abs().max().to(torch.float64), utility.new_tensor(1.))
    winning = utility[:, None] - utility[None, :] > tolerance
    if not winning.any(): return scores.sum() * 0
    forecast = weights @ scores.to(torch.float64)
    margins = forecast[:, None] - forecast[None, :]
    return F.softplus(-margins[winning]).mean()
