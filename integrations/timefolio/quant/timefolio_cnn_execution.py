"""Causal CNN ranking snapshots for execution between scheduled rebalances.

Only scores are held fixed. The account engine still evaluates current cash,
eligibility, caps, order budgets and fills on every order-decision date.
"""
import numpy as np


def freeze_rankings(scores, dates, start, end, *, rebalance):
    """Align score snapshots with the existing replay's decision-day cadence.

    A decision on date index d consumes scores from d-1. Deferred orders keep
    that snapshot until the next scheduled rebalance; missing scores remain
    missing. The returned anchor index records when each used score was known.
    """
    scores=np.asarray(scores)
    dates=list(dates)
    if (scores.ndim!=2 or scores.shape[1]!=len(dates)
            or not np.issubdtype(scores.dtype,np.floating) or np.isinf(scores).any()
            or dates!=sorted(set(dates)) or start>end
            or type(rebalance) is not int or rebalance<1):
        raise ValueError('Ordered matching dates, finite-or-missing scores and positive cadence required')
    days=[d for d,date in enumerate(dates) if start<=date<=end and d>0]
    if not days:
        raise ValueError('No order-decision dates in the requested interval')
    result=scores.copy()
    anchors=np.full(len(dates),-1,dtype=np.int64)
    for k,day in enumerate(days):
        if k%rebalance==0:
            anchor=day-1
        result[:,day-1]=scores[:,anchor]
        anchors[day-1]=anchor
    return result,anchors
