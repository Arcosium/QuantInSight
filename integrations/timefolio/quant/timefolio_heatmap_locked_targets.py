"""Reserve unsellable entitlement quantities before trimming tradable targets."""
import numpy as np


def reserve_locked(desired,qty,locked,prices,planning_good,nav,sectors,sector_caps,small,
                   stock_caps,scores,codes,*,gross=.8):
    """Use planning-time information only; no future fills, volume or close.

    Cancel proposed purchases first, then reduce low-ranked tradable positions.
    Locked quantities are a hard floor, so unavoidable residual excess remains
    visible. The caller retains its ordinary fill/participation/order constraints.
    """
    arrays=[np.asarray(v) for v in [desired,qty,locked,prices,planning_good,sectors,sector_caps,small,stock_caps,scores,codes]]
    desired,qty,locked,prices,good,sectors,scaps,small,icaps,scores,codes=arrays
    n=len(qty)
    if any(a.shape!=(n,) for a in arrays):raise ValueError('Matching one-dimensional arrays required')
    if any(not np.issubdtype(a.dtype,np.integer) for a in [desired,qty,locked]):raise ValueError('Integer share quantities required')
    if (np.any(desired<0) or np.any(qty<0) or np.any(locked<0) or np.any(locked>qty)
        or not np.isfinite(nav) or nav<=0 or not 0<=gross<=1):raise ValueError('Invalid inventory or NAV')
    if not np.isfinite(prices).all() or np.any(prices<0):raise ValueError('Finite nonnegative planning marks required')
    if not np.isfinite(scaps).all() or not np.isfinite(icaps).all() or np.any(scaps<=0) or np.any(icaps<=0):
        raise ValueError('Finite positive limits required')
    good=good.astype(bool)&(prices>0);small=small.astype(bool)
    floor=np.where(good,locked,qty).astype(np.int64)
    out=np.maximum(desired,floor).astype(np.int64)
    if not np.any(locked>desired):return desired.copy()
    order=np.lexsort((codes,np.nan_to_num(scores,nan=-np.inf)))
    def trim(members,limit):
        excess=float(out[members]@prices[members]-limit)
        if excess<=1:return
        # Removing unfilled purchase targets avoids unnecessary sales.
        for lower in [np.maximum(qty,floor),floor]:
            for i in order:
                if not members[i] or not good[i] or out[i]<=lower[i]:continue
                cut=min(int(out[i]-lower[i]),max(0,int(np.ceil(excess/prices[i]))))
                out[i]-=cut;excess-=cut*prices[i]
                if excess<=1:return
    for i in range(n):
        if good[i]:out[i]=max(floor[i],min(out[i],int(np.floor(icaps[i]*nav/prices[i]))))
    for sector in np.unique(sectors):
        members=sectors==sector
        if not np.all(scaps[members]==scaps[members][0]):raise ValueError('Sector limits disagree within group')
        trim(members,.95*float(scaps[members][0])*nav)
    trim(small,.285*nav);trim(np.ones(n,bool),gross*nav)
    assert np.all(out>=floor) and np.all(out<=np.maximum(desired,floor))
    return out
