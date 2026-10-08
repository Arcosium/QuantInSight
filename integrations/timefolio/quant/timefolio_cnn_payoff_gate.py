"""Chronological ridge gates for matched, cost-adjusted policy-payoff targets.

Only matured training labels enter a fit. Short cash-start policy payoffs are
research targets; final selection still requires a full constrained account.
"""
from __future__ import annotations
import numpy as np


def walk_payoffs(features, targets, signals, label_available, folds, *, days,
                 feature_available=None, minimum_rows=60, alpha=10.):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    s, end = np.asarray(signals), np.asarray(label_available)
    available = s if feature_available is None else np.asarray(feature_available)
    if (x.ndim != 2 or x.shape[1] < 1 or y.ndim != 2 or y.shape[1] < 1
            or len(x) != len(y) or s.shape != (len(x),) or end.shape != s.shape
            or available.shape != s.shape or any(not np.issubdtype(a.dtype, np.integer) for a in [s,end,available])
            or not np.isfinite(x).all() or np.isinf(y).any() or not np.all(np.diff(s)>0)
            or type(days) is not int or days < 1 or np.any(s<0) or np.any(s>=days)
            or np.any(end<=s) or np.any(available>s)
            or type(minimum_rows) is not int or minimum_rows<2 or not np.isfinite(alpha) or alpha<=0):
        raise ValueError('Ordered causal features, explicit maturity and finite bounded configuration required')
    expected=np.full((days,y.shape[1]),np.nan); cold=np.zeros(days,bool)
    covered=np.zeros(days,bool); fits=[]; finite=np.isfinite(y).all(axis=1)
    for fold in folds:
        cutoff,stop=fold['test_start'],fold['test_end']
        if (type(cutoff) is not int or type(stop) is not int or not 0<=cutoff<stop<=days
                or covered[cutoff:stop].any()):
            raise ValueError('Disjoint valid forecast folds required')
        covered[cutoff:stop]=True
        forecast=(s>=cutoff)&(s<stop)
        if not forecast.any():
            continue
        # All target columns and feature variants use exactly the same rows.
        train=(s<cutoff)&(end<cutoff)&finite
        record=dict(fold=fold['id'],fit_cutoff=cutoff,matured_rows=int(train.sum()),
                    training_signals=s[train].tolist(),last_label_available=int(end[train].max()) if train.any() else None,
                    forecast_signals=s[forecast].tolist(),cold_start=int(train.sum())<minimum_rows)
        if record['cold_start']:
            cold[s[forecast]]=True
            record['fallback']='80percent gross until60matured matched target rows'
        else:
            mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-8)
            z=(x[train]-mean)/scale;intercept=y[train].mean(0)
            beta=np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@(y[train]-intercept))
            expected[s[forecast]]=(x[forecast]-mean)/scale@beta+intercept
            record.update(mean=mean.tolist(),scale=scale.tolist(),coefficient=beta.tolist(),intercept=intercept.tolist())
        fits.append(record)
    if not covered[s].all():
        raise ValueError('Every feature row must belong to a registered forecast fold')
    gross=np.where(np.isnan(expected)|(expected>0),.8,.2)
    return dict(expected_net=expected,gross=gross,cold_start=cold,fits=fits)
