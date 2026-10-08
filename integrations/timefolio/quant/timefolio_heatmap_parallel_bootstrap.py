"""Retain the full max-t family while bounding bootstrap storage by columns.

Every column sees the same registered circular-block draws. Standard errors and
marginal p-values are computed before discarding each bootstrap block; per-draw
maxima are retained across all columns for the joint correction.
"""
from concurrent.futures import ThreadPoolExecutor
import operator
import numpy as np
from quant.timefolio_heatmap_walkforward_eval import block_indices


def family_bootstrap(differences, *, block=5, draws=4000, seed=57,
                     column_batch=1024, scratch_bytes=256*1024**2,
                     bootstrap_bytes=64*1024**2, workers=8):
    workers = operator.index(workers)
    if not 1 <= workers <= 8:
        raise ValueError("One to eight remote bootstrap workers required")
    a = np.asarray(differences, float)
    if a.ndim != 2 or len(a) < 2 or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Finite day-by-strategy array required')
    block, draws, column_batch = map(operator.index, (block, draws, column_batch))
    if block < 1 or draws < 2 or column_batch < 1:
        raise ValueError('Positive block/column batch and at least two draws required')
    budget_columns = min(bootstrap_bytes // (draws*8), scratch_bytes // (len(a)*8))
    minimum_columns = min(2, a.shape[1])
    if budget_columns < minimum_columns:
        raise ValueError('Memory budgets cannot preserve the original reduction layout')
    columns = min(column_batch, budget_columns)
    mean = a.mean(0); se = np.empty(a.shape[1]); marginal = np.empty(a.shape[1])
    maxima = np.full(draws, -np.inf)
    rng = np.random.default_rng(seed)
    indices = [block_indices(len(a), block, min(100, draws-offset), rng)
               for offset in range(0, draws, 100)]
    def column(start):
        end = min(start+columns, a.shape[1])
        width = end-start
        # A one-column tail would switch NumPy to contiguous pairwise
        # reductions. Duplicate it physically to retain the original
        # multi-column reduction order; exclude the padding from inference.
        storage_width = max(width, minimum_columns)
        centered = np.empty((len(a), storage_width))
        centered[:, :width] = a[:, start:end] - mean[start:end]
        if storage_width > width: centered[:, width:] = centered[:, width-1:width]
        boot = np.empty((draws, storage_width))
        batch = max(1, min(100, scratch_bytes // centered.nbytes))
        offset = 0
        for idx in indices:
            for j in range(0, len(idx), batch):
                selected = idx[j:j+batch]
                boot[offset+j:offset+j+len(selected)] = centered[selected].mean(1)
            offset += len(idx)
        scale = np.maximum(boot.std(0, ddof=1), 1e-12)[:width]
        observed = mean[start:end] / scale
        standardized = boot[:, :width] / scale
        local_maxima = standardized.max(axis=1)
        local_marginal = (1+(standardized >= observed).sum(0))/(draws+1)
        return start, end, scale, local_marginal, local_maxima
    # Each task retains the exact original column width, padding and reduction
    # order. Only independent columns execute concurrently; RNG draws are shared.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, end, scale, local_marginal, local_maxima in executor.map(
                column, range(0, a.shape[1], columns)):
            se[start:end] = scale
            marginal[start:end] = local_marginal
            maxima = np.maximum(maxima, local_maxima)
    observed = mean/se; adjusted = np.empty(a.shape[1])
    for start in range(0, a.shape[1], columns):
        end = min(start+columns, a.shape[1])
        adjusted[start:end] = (1+(maxima[:, None] >= observed[start:end]).sum(0))/(draws+1)
    critical = np.quantile(maxima, .95)
    return dict(mean=mean, standard_error=se, adjusted_p=adjusted, marginal_p=marginal,
                simultaneous_lower95=mean-critical*se, critical_max_t=float(critical),
                block=block, draws=draws)

PARENT_SHA256 = '565e141d53a0e22ca97456983a9a038b7b56e917bbaac697e533eca94e1cce2c'
