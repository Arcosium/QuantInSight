"""Causal score aggregation and realized absolute-return calibration.

Four fixed development variants motivated by the crypto/KRX diagnostic.
All calibration labels must have matured by the signal close. No shorts.
"""
import numpy as np
from quant.timefolio_heatmap_transfer_diagnostics import ranks
from quant.timefolio_heatmap_fleet_accounts import policy_grid, account_id
from quant.timefolio_heatmap_replay import BUY_FEE, SELL_FEE

VARIANTS = ['smooth5','smooth10','cal20','cal60']
MEMBERS = ['seed17','seed29','seed43','ensemble']


def percentiles(score, eligible):
    out=np.full(score.shape,np.nan,dtype=float)
    for d in range(score.shape[1]):
        ids=np.flatnonzero(eligible[:,d] & np.isfinite(score[:,d]))
        if len(ids):out[ids,d]=(ranks(score[ids,d])+1)/len(ids)
    return out


def observed_net_labels(panel, horizon=5):
    """Labels are stored by signal date; users must check d+h <= signal close."""
    close=np.asarray(panel['close'],float);entry=np.asarray(panel['exec_price'],float)
    y=np.full(close.shape,np.nan)
    cost=(1-.0005)*(1-SELL_FEE)/((1+.0005)*(1+BUY_FEE))
    for d in range(close.shape[1]-horizon):
        valid=(np.isfinite(entry[:,d+1]) & (entry[:,d+1]>0)
            & np.isfinite(close[:,d+horizon]) & (close[:,d+horizon]>0)
            & (panel['exec_count'][:,d+1]>=25)
            & np.all(np.abs(panel['split'][:,d+1:d+horizon+1]-1)<=.002,axis=1))
        y[valid,d]=close[valid,d+horizon]/entry[valid,d+1]*cost-1
    return y


def transform(score, panel, dates, variant):
    if variant not in VARIANTS:raise ValueError('Unregistered transfer variant')
    score=np.asarray(score,float);eligible=np.asarray(panel['eligible'],bool)
    if score.shape!=eligible.shape:raise ValueError('Score/panel axes differ')
    pct=percentiles(score,eligible);out=np.full_like(pct,np.nan)
    n=score.shape[1];receipt=[]
    if variant.startswith('smooth'):
        width=int(variant[6:])
        for d in range(n):
            x=pct[:,max(0,d-width+1):d+1];count=np.isfinite(x).sum(axis=1)
            valid=np.isfinite(pct[:,d]) & (count>0)
            out[valid,d]=np.nansum(x[valid],axis=1)/count[valid]
        return out,None,dict(variant=variant,uses_returns=False,maximum_feature_date='signal close')
    lookback=int(variant[3:]);horizon=5
    labels=observed_net_labels(panel,horizon)
    first=next(i for i,d in enumerate(dates) if d>='20260101')
    bins=np.minimum(9,np.floor(np.nan_to_num(pct,nan=0.)*10).astype(int))
    daily=np.full((n,10),np.nan);overall=np.full(n,np.nan)
    for d in range(first,n-horizon):
        valid=np.isfinite(pct[:,d]) & np.isfinite(labels[:,d])
        if valid.sum()<20:continue
        overall[d]=np.mean(labels[valid,d])
        for b in range(10):
            v=valid & (bins[:,d]==b)
            if v.any():daily[d,b]=np.mean(labels[v,d])
    gross=np.full(n,.6)
    for d in range(n):
        valid=np.isfinite(pct[:,d]);out[valid,d]=pct[valid,d]
        last=d-horizon;start=max(first,last-lookback+1)
        observed=[i for i in range(start,last+1) if np.isfinite(overall[i])]
        if len(observed)<20:
            receipt.append(dict(date=dates[d],fitted=False,observed_dates=len(observed)));continue
        assert max(observed)+horizon<=d
        prior=np.mean(overall[observed]);values=daily[observed];count=np.isfinite(values).sum(axis=0)
        # Five day-equivalents of shrinkage, fixed before new account outcomes.
        means=(np.nansum(values,axis=0)+5*prior)/(count+5)
        expected=means[bins[valid,d]]
        out[valid,d]=expected+1e-8*pct[valid,d]
        best=np.sort(expected)[-12:]
        gross[d]=.6 if len(best)>=12 and best.mean()>0 else .2
        receipt.append(dict(date=dates[d],fitted=True,observed_dates=len(observed),
            first_label_signal=dates[min(observed)],last_label_signal=dates[max(observed)],
            last_observed_label_close=dates[max(observed)+horizon],
            prior_net_return=float(prior),decile_net_returns=means.tolist(),gross=float(gross[d])))
    return out,gross,dict(variant=variant,horizon=5,lookback=lookback,minimum_matured_dates=20,
        shrinkage_day_equivalents=5,receipts=receipt,missing_or_action_labels_excluded=True,
        caveat='Exploratory overlapping-label calibration, not independent confirmation.')


def identity(model,variant,policy):
    return 'transfer_'+variant+'_'+account_id(model,policy)


def hypotheses(cases):
    family=[]
    for case in cases:
        for member in MEMBERS:
            for variant in VARIANTS:
                for policy in policy_grid():
                    model=f'flt_{case}_trained_{member}'
                    key=identity(model,variant,policy)
                    untrained=identity(f'flt_{case}_untrained_{member}',variant,policy)
                    family.extend([(key,'cash',None),(key,'matching_untrained_transfer',untrained),
                        (key,'same_transfer_nonimage',identity('online_nonimage',variant,policy)),
                        (key,'unchanged_original',account_id(model,policy)),(untrained,'cash',None)])
    family.extend((identity('online_nonimage',v,p),'cash',None) for v in VARIANTS for p in policy_grid())
    assert len(family)==len({(key,label) for key,label,_ in family})
    return family
