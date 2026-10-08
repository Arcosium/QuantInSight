"""Fixed semantic BCE probability gate for already audited OOF forecasts.

The ranker chooses a basket; a separate BCE model supplies absolute logits for
that basket. These are uncalibrated target-positive probabilities, not expected
cash returns. The account engine applies signal-date exposure at the next open.
"""
import numpy as np


def absolute_basket_gate(ranking, logits, security_keys, *, probability_objective,
                         top_k=10, threshold=.5, gross=.8, defensive_floor=.2):
    """Return same-date basket probabilities and two requested exposure paths.

    Missing whole dates remain NaN; partial model membership fails closed.
    No cross-sectional centering, future normalization, prices or labels enter.
    The caller must validate dated axes, model receipts and OOF availability.
    """
    scores, raw, keys = map(np.asarray, [ranking, logits, security_keys])
    if probability_objective != 'bce':
        raise ValueError('Only audited BCE logits have the required probability meaning')
    if (scores.ndim != 2 or raw.shape != scores.shape or not all(scores.shape)
            or any(not np.issubdtype(a.dtype, np.floating) or np.isinf(a).any()
                   for a in [scores, raw])
            or keys.shape != (scores.shape[1],) or len(np.unique(keys)) != len(keys)):
        raise ValueError('Matching floating forecasts and unique security keys required')
    if (type(top_k) is not int or not 1 <= top_k <= scores.shape[1]
            or not all(np.isfinite(v) for v in [threshold, gross, defensive_floor])
            or not 0 < threshold < 1 or not 0 <= defensive_floor < gross <= 1):
        raise ValueError('Invalid basket or exposure policy')
    membership = np.isfinite(scores)
    if not np.array_equal(membership, np.isfinite(raw)):
        raise ValueError('Ranker and BCE must predict exactly the same observations')
    probability = np.full(scores.shape[0], np.nan)
    count = membership.sum(axis=1)
    if np.any((count > 0) & (count < top_k)):
        raise ValueError('Incomplete basket forecast date')
    for day in np.flatnonzero(count):
        columns = np.flatnonzero(membership[day])
        top = columns[np.lexsort((keys[columns], -scores[day, columns]))[:top_k]]
        z = raw[day, top].astype(np.float64)
        # Both branches are bounded; do not evaluate exp(-z) at large negative z.
        positive = z >= 0
        p = np.empty(len(z))
        p[positive] = 1 / (1 + np.exp(-z[positive]))
        e = np.exp(z[~positive]); p[~positive] = e / (1 + e)
        probability[day] = p.mean()
    cash = np.where(count, np.where(probability > threshold, gross, 0.), np.nan)
    return dict(mean_probability=probability, gross_cash=cash,
                gross_floor20=np.maximum(cash, defensive_floor),
                forecast_count=count)
