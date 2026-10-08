"""Descriptive distances to reported crypto metrics, not a cross-market test."""
import math
import numpy as np


def nav_metrics(nav, *, initial=1e9, annual_days=252):
    values = np.asarray(nav, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('At least two finite positive closing NAV values required')
    if not np.isfinite(initial) or initial <= 0 or not isinstance(annual_days, int) or annual_days <= 0:
        raise ValueError('Positive initial capital and annual observation count required')
    returns = values / np.r_[initial, values[:-1]] - 1
    std = float(returns.std(ddof=1)); defined = std > 1e-12
    peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
    final = float(values[-1] / initial)
    return dict(days=len(values), annual_days=annual_days, final=final, net_return=final - 1,
                sharpe=float(returns.mean() / std * math.sqrt(annual_days)) if defined else 0.,
                sharpe_defined=defined, mdd=float(np.min(values / peaks - 1)),
                annualized_growth=math.expm1(math.log(final) * annual_days / len(values)))


def reported_reference(row, *, annual_days=365):
    """Table MDD is percent; Sharpe/MDD/final are rounded to 2/1/4 decimals."""
    sharpe, mdd, final, days = (float(row[k]) for k in ['sharpe', 'mdd', 'final', 'days'])
    if not all(math.isfinite(v) for v in [sharpe, mdd, final, days]) or sharpe <= 0 or mdd >= 0 or final <= 1 or days < 2 or days != int(days):
        raise ValueError('Positive-return benchmark with finite reported metrics required')
    growth = lambda ratio: math.expm1(math.log(ratio) * annual_days / days)
    return dict(sharpe=sharpe, mdd=mdd / 100, final=final, days=int(days), annual_days=annual_days,
                annualized_growth=growth(final), growth_is_derived_not_paper_reported=True,
                sharpe_upper_rounding_bound=sharpe + .005,
                mdd_stricter_rounding_bound=(mdd + .05) / 100,
                growth_upper_rounding_bound=growth(final + .00005))


def distance(metrics, reference, *, near_fraction=.90):
    if not 0 < near_fraction < 1:
        raise ValueError('Near fraction must be strictly between zero and one')
    s, g, d = metrics['sharpe'], metrics['annualized_growth'], metrics['mdd']
    rs, rg, rd = reference['sharpe'], reference['annualized_growth'], reference['mdd']
    defined = bool(metrics['sharpe_defined']); tol = 1e-12
    return dict(sharpe_gap=s-rs, growth_gap=g-rg, mdd_gap=d-rd,
        sharpe_fraction=s/rs, growth_fraction=g/rg, drawdown_multiple=abs(d)/abs(rd),
        sharpe_near=defined and s >= near_fraction*rs-tol,
        all_metrics_near=defined and s >= near_fraction*rs-tol and g >= near_fraction*rg-tol and abs(d) <= (2-near_fraction)*abs(rd)+tol,
        nominal_joint_exceed=defined and s > rs+tol and g > rg+tol and d >= rd-tol,
        conservative_joint_exceed=defined and s > reference['sharpe_upper_rounding_bound']+tol
            and g > reference['growth_upper_rounding_bound']+tol and d >= reference['mdd_stricter_rounding_bound']-tol)
