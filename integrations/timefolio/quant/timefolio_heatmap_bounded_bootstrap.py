"""Same max-t arithmetic and random draws, with bounded resampling scratch."""
import numpy as np
from quant.timefolio_heatmap_walkforward_eval import block_indices


def family_bootstrap(differences,*,block=5,draws=4000,seed=57,scratch_bytes=256*1024**2):
    a=np.asarray(differences,float)
    if a.ndim!=2 or len(a)<2 or not a.shape[1] or not np.isfinite(a).all():raise ValueError('Finite day-by-strategy array required')
    bytes_per_draw=a.size*a.dtype.itemsize
    if scratch_bytes<bytes_per_draw:raise ValueError('Scratch ceiling cannot hold one resampled draw')
    if draws*a.shape[1]*8>2*1024**3:raise ValueError('Bootstrap matrix exceeds registered2GiB ceiling')
    batch=max(1,min(100,scratch_bytes//bytes_per_draw));rng=np.random.default_rng(seed)
    mean=a.mean(0);centered=a-mean;boot=np.empty((draws,a.shape[1]))
    for offset in range(0,draws,100):
        # Preserve the original RNG call shape and order, then split only the
        # independent sample-mean calculations along the outer draw dimension.
        idx=block_indices(len(a),block,min(100,draws-offset),rng)
        for j in range(0,len(idx),batch):
            selected=idx[j:j+batch];boot[offset+j:offset+j+len(selected)]=centered[selected].mean(1)
    se=np.maximum(boot.std(0,ddof=1),1e-12);maxima=np.max(boot/se,axis=1);observed=mean/se
    adjusted=(1+(maxima[:,None]>=observed).sum(0))/(draws+1)
    marginal=(1+(boot/se>=observed).sum(0))/(draws+1);critical=np.quantile(maxima,.95)
    return dict(mean=mean,standard_error=se,adjusted_p=adjusted,marginal_p=marginal,
        simultaneous_lower95=mean-critical*se,critical_max_t=float(critical),block=block,draws=draws)
