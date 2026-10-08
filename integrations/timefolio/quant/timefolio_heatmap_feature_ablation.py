"""Outcome-blind row ablations and an explicit daily-feature availability mask."""
import numpy as np
from quant.timefolio_heatmap_daily_history import DAILY_ROWS,daily_values

GROUPS=dict(price=list(range(8))+[28,29],flow=list(range(8,16))+[27,31],
    momentum=list(range(16,22)),context=[22,23],constraints=[24,25,26],time=[30])
VARIANTS=['full','price_only','flow_only','daily_only','no_price','no_flow','no_momentum','no_context',
    'no_constraints','no_ranks','last_day','row_center','mask_constant','mask_available']


def availability(panel,ci,di):
    ci,di=np.asarray(ci),np.asarray(di)
    if ci.ndim!=1 or di.shape!=ci.shape or np.any(di<4):raise ValueError('Matching five-day sample axes required')
    values=daily_values(panel,ci,di[:,None]+np.arange(-4,1))
    mask=np.full((len(ci),32,65),255,np.uint8)
    mask[:,DAILY_ROWS]=np.repeat(np.isfinite(values).astype(np.uint8)*255,13,axis=2)
    return mask


def transform(history,variant,mask=None):
    """Keep shape fixed; removed values use the original uint8 neutral code128."""
    if history.dtype!=np.uint8 or history.ndim!=3 or history.shape[1:]!=(32,65):raise ValueError('uint8 Nx32x65 required')
    if variant not in VARIANTS:raise ValueError('Unregistered representation')
    out=history.copy()
    if variant in ['price_only','flow_only','daily_only']:
        keep=(GROUPS[variant.split('_')[0]] if variant!='daily_only' else list(DAILY_ROWS))+[30]
        out[:,[r for r in range(32) if r not in keep]]=128
    elif variant.startswith('no_'):
        rows=[15,20,21] if variant=='no_ranks' else GROUPS[variant[3:]]
        out[:,rows]=128
    elif variant=='last_day':out[:,:,:52]=128
    elif variant=='row_center':
        values=out.astype(np.float64)
        # Center exact stored integer codes before scaling; constant rows map
        # exactly to code128 without float32 cancellation around the half tie.
        out=np.rint(np.clip(values-values.mean(axis=2,keepdims=True)+127.5,0,255)).astype(np.uint8)
    if variant in ['mask_constant','mask_available']:
        if variant=='mask_constant':extra=np.full_like(out,255)
        else:
            if mask is None or mask.shape!=out.shape or mask.dtype!=np.uint8 or not np.isin(mask,[0,255]).all():
                raise ValueError('Binary daily-availability image required')
            extra=mask
        return np.stack([out,extra],axis=1)
    return out[:,None]
