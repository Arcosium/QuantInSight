"""Single-entry position-PnL labels; no downloads, training or broker actions.

Future execution data are labels only. This isolated-position surrogate does
not model competition for sector headroom, portfolio order slots or retention.
The economic labels and signal-time inference membership remain unchanged.
"""
from __future__ import annotations

import numpy as np


def execution_targets(panel, codes, signal, keys, end, economic_returns, *, horizon=20):
    """Return matched closing-mark and executable PnL targets for an8% sleeve.

    Start with1bn cash, make one next-session buy request, and mark through
    signal+H close without a sale or retry. Unfilled cash earns zero. Retain
    the parent's stricter signal+H+1 label-availability index and finite mask.
    Panels with corporate-action ratios are refused, not silently ignored.
    """
    s, k, e, r = map(np.asarray, (signal, keys, end, economic_returns))
    shape = np.shape(panel['close'])
    required = ('o', 'close', 'eligible', 'sector_cap', 'market_cap', 'split',
                'exec_price', 'exec_volume', 'exec_count', 'exec_high', 'exec_low')
    if (len(shape) != 2 or len(codes) != shape[0] or s.ndim != 1
            or any(a.shape != s.shape for a in (k, e, r))
            or any(not np.issubdtype(a.dtype, np.integer) for a in (s, k, e))
            or type(horizon) is not int or horizon < 1
            or any(np.shape(panel[name]) != shape for name in required)
            or np.any(s < 0) or np.any(s >= shape[1])
            or np.any(k < 0) or np.any(k >= shape[0])
            or np.any(e != s+horizon+1) or np.isinf(r).any()
            or np.any(np.isfinite(r) & (e >= shape[1]))):
        raise ValueError('Registered matched label axes required')
    if not np.all(np.asarray(panel['split']) == 1):
        raise ValueError('Action-adjusted position labels require a separate action ledger')
    if 'trade_allowed' in panel and np.shape(panel['trade_allowed']) != shape:
        raise ValueError('Invalid trade admission axes')
    nav, weight, gross, fee, slip, participation = 1e9, .08, .8, .001, .0005, .05
    out = {name: np.full(len(s), np.nan, dtype=np.float64) for name in
           ('close_gross', 'position_utility', 'account_return', 'requested_qty',
            'filled_qty', 'entry_cash_spent', 'terminal_mark')}
    codes = np.asarray(codes)
    ids = np.flatnonzero(np.isfinite(r))
    for lo in range(0, len(ids), 2048):
        ix = ids[lo:lo+2048]; j, t = k[ix], s[ix]; d = t+1
        previous = np.asarray(panel['close'][j, t], dtype=float)
        planned = np.asarray(panel['o'][j, d], dtype=float).copy()
        invalid = ~np.isfinite(planned) | (planned <= 0)
        planned[invalid] = previous[invalid]
        # Original finite open/open labels should provide valid entry prices.
        if not (np.isfinite(planned) & (planned > 0)).all():
            raise ValueError('Finite economic label lacks an entry planning price')
        opening = np.asarray(panel['exec_price'][j, d], dtype=float)
        good = np.isfinite(opening) & (opening > 0) & (panel['exec_count'][j, d] >= 25)
        admitted = np.asarray(panel['eligible'][j, t], dtype=bool).copy()
        if 'trade_allowed' in panel:
            admitted &= np.asarray(panel['trade_allowed'][j, d], dtype=bool)
        cap = np.asarray(panel['sector_cap'][j, t], dtype=float)
        old_cap = np.asarray(panel['market_cap'][j, t], dtype=float)
        # Twice a sector market weight can exceed100%; total investment and
        # individual-position limits independently bound actual exposure.
        if not (np.isfinite(cap) & (cap > 0) & (cap <= 2)).all():
            raise ValueError('Invalid signal-time sector cap')
        statutory = np.where(codes[j] == '005930', .4,
                             np.where(codes[j] == '000660', .3, .15))
        w = np.minimum(np.minimum(weight, statutory), np.minimum(gross, .95*cap))
        w = np.where(old_cap < 1e12, np.minimum(w, .285), w)
        w = np.where(admitted & (w > .005), w, 0.)
        requested = np.floor(w*nav/planned)
        volume = np.asarray(panel['exec_volume'][j, d], dtype=float)
        if np.isinf(volume).any() or np.any(volume[np.isfinite(volume)] < 0):
            raise ValueError('Invalid execution-window volume')
        capacity = np.floor(np.nan_to_num(volume)*participation)
        locked = np.isclose(panel['exec_high'][j, d], panel['exec_low'][j, d], rtol=0, atol=.001)
        with np.errstate(divide='ignore', invalid='ignore'):
            up = locked & (opening/previous >= 1.29)
        relative = np.divide(opening, previous, out=np.ones(len(ix)), where=previous > 0)
        cap_now = old_cap*relative
        fillable = good & ~up & np.isfinite(cap_now) & (cap_now >= 1e11)
        # A safe placeholder lets blocked requests produce a zero/cash target.
        mark = np.where(good, opening, 1.)
        cost = mark*(1+slip)*(1+fee)
        loss = cost-mark
        q = np.minimum(requested, capacity)
        for ceiling in (statutory, cap, np.full(len(ix), gross)):
            q = np.minimum(q, ceiling*nav/(mark+ceiling*loss))
        q = np.minimum(q, nav/cost)
        small_limit = np.where(cap_now < 1e12, .3*nav/(mark+.3*loss), nav/loss)
        q = np.minimum(q, small_limit)
        q = np.where(fillable, np.maximum(0, np.floor(q)), 0.)
        spent = q*cost
        terminal = np.zeros(len(ix))
        # Same marks as the replay: valid execution VWAP, then valid close;
        # otherwise carry the last observed mark. No look beyond signal+H.
        for offset in range(1, horizon+1):
            day = t+offset
            px = np.asarray(panel['exec_price'][j, day], dtype=float)
            valid = np.isfinite(px) & (px > 0) & (panel['exec_count'][j, day] >= 25)
            terminal[valid] = px[valid]
            close = np.asarray(panel['close'][j, day], dtype=float)
            valid = np.isfinite(close) & (close > 0)
            terminal[valid] = close[valid]
        # No known price at all cannot define the closing-price control.
        if np.any(terminal <= 0):
            raise ValueError('Finite economic label lacks a holding-period mark')
        pnl = q*terminal-spent
        values = dict(close_gross=terminal/planned-1,
                      position_utility=pnl/(weight*nav), account_return=pnl/nav,
                      requested_qty=requested, filled_qty=q, entry_cash_spent=spent,
                      terminal_mark=terminal)
        for name, value in values.items():
            out[name][ix] = value
    return out
