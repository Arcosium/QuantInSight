"""Equal-weight seed ensemble with exact common forecast membership.

Scores have axes (date, security). Only the same date's forecasts enter its
percentile ranks; labels, realized returns and cross-date normalization do not.
"""
import numpy as np


def mean_rank_percentiles(score_matrices):
    """Average tied midrank percentiles, preserving missing forecasts as NaN.

    Percentiles use (one-based average rank - 0.5) / count. A singleton or
    fully tied cross-section is 0.5. A missing seed forecast fails closed.
    """
    matrices=[np.asarray(scores) for scores in score_matrices]
    if len(matrices)<2:
        raise ValueError('At least two complete seed forecast matrices required')
    shape=matrices[0].shape
    if (len(shape)!=2 or not all(shape) or any(
            matrix.shape!=shape or not np.issubdtype(matrix.dtype,np.floating)
            or np.isinf(matrix).any() for matrix in matrices)):
        raise ValueError('Matching floating date/security matrices required')
    membership=np.isfinite(matrices[0])
    if any(not np.array_equal(np.isfinite(matrix),membership) for matrix in matrices[1:]):
        raise ValueError('Every seed must predict exactly the same observations')
    combined=np.full(shape,np.nan,dtype=np.float32)
    for day in range(shape[0]):
        columns=np.flatnonzero(membership[day])
        count=len(columns)
        if not count:
            continue
        total=np.zeros(count,dtype=np.float64)
        for matrix in matrices:
            values=matrix[day,columns]
            order=np.argsort(values,kind='stable')
            ordered=values[order]
            bounds=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,count]
            ranks=np.empty(count,dtype=np.float64)
            ranks[order]=np.repeat((bounds[:-1]+bounds[1:])/(2.*count),np.diff(bounds))
            total+=ranks
        combined[day,columns]=total/len(matrices)
    return combined
