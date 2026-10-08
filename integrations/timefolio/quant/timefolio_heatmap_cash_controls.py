"""Finite, causal cash-exposure controls for fixed long-only heatmap scores.

These are exploratory controls, not calibrated return probabilities. Output at
column t uses information through that close; the frozen replay consumes t at
the next session's execution window. No parameters are estimated on OOS returns.
"""
import numpy as np


def _panel(panel):
    close = np.asarray(panel['close'], dtype=float)
    eligible = np.asarray(panel['eligible'], dtype=bool)
    split = np.asarray(panel['split'], dtype=float)
    if close.ndim != 2 or eligible.shape != close.shape or split.shape != close.shape:
        raise ValueError('Aligned stock by session close, eligibility and split arrays required')
    if not np.isfinite(split).all() or np.any(split <= 0):
        raise ValueError('Positive known share-adjustment ratios required')
    return close, eligible, split


def price_schedules(panel):
    close, eligible, split = _panel(panel)
    n = close.shape[1]
    daily = np.zeros(n); breadth = np.zeros(n)
    positive = np.zeros_like(close, dtype=bool)
    observed = np.isfinite(close) & (close > 0)
    # Share-adjusted total price changes include the current day's known split.
    returns = np.full(close.shape, np.nan)
    for t in range(1, n):
        valid = observed[:, t] & observed[:, t-1]
        returns[valid, t] = close[valid, t] * split[valid, t] / close[valid, t-1] - 1
        members = valid & eligible[:, t-1]
        if members.any(): daily[t] = returns[members, t].mean()
        if t >= 20:
            history = returns[:, t-19:t+1]
            known = eligible[:, t] & np.isfinite(history).all(axis=1)
            if known.any():
                positive[:, t] = np.prod(1 + history, axis=1) > 1
                breadth[t] = positive[known, t].mean()
    trend = np.full(n, .2); breadth_gross = np.full(n, .2); volatility = np.full(n, .2)
    for t in range(20, n):
        recent = daily[t-19:t+1]
        gain20 = np.prod(1 + recent) - 1
        sigma = recent.std(ddof=1) * np.sqrt(252)
        volatility[t] = np.clip(.6 * .15 / max(sigma, .05), .15, .6)
        if gain20 > 0 and breadth[t] > .5: breadth_gross[t] = .6
        if t >= 60 and gain20 > 0 and np.prod(1 + daily[t-59:t+1]) > 1:
            trend[t] = .6
    return dict(trend20_60=trend, breadth20=breadth_gross, volatility20=volatility)


def schedules(panel, absolute_mse_scores):
    """Seven fixed controls, plus a separate unchanged no-schedule baseline."""
    close, eligible, _ = _panel(panel)
    scores = np.asarray(absolute_mse_scores, dtype=float)
    if scores.shape != close.shape:
        raise ValueError('Forecast scores must share the market date and stock axes')
    price = price_schedules(panel)
    confidence = {name: np.full(close.shape[1], .2) for name in ['forecast0', 'forecast005']}
    for t in range(close.shape[1]):
        values = scores[eligible[:, t] & np.isfinite(scores[:, t]), t]
        if len(values) < 12: continue
        top = np.sort(values)[-12:].mean()
        if top > 0: confidence['forecast0'][t] = .6
        if top > .05: confidence['forecast005'][t] = .6
    result = dict(price, **confidence,
        trend_forecast0=np.minimum(price['trend20_60'], confidence['forecast0']),
        breadth_forecast005=np.minimum(price['breadth20'], confidence['forecast005']))
    assert len(result) == 7
    assert all(np.isfinite(v).all() and np.all((v >= .15) & (v <= .6)) for v in result.values())
    return result
