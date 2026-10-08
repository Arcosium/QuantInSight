"""Opening-price overlays for forecasts already frozen at an earlier close.

Output columns are decision dates, not forecast dates. Only the current open
and previous close/volatility are added; current execution data is never read.
"""
import numpy as np
from quant.timefolio_heatmap_transfer_diagnostics import ranks
from quant.timefolio_heatmap_fleet_accounts import policy_grid, account_id

COEFFICIENTS={'fade25':-.25,'fade50':-.5,'confirm25':.25,'confirm50':.5}
VARIANTS=list(COEFFICIENTS)
MEMBERS=['seed17','seed29','seed43','ensemble']


def transform(held,panel,dates,variant):
    coefficient=0. if variant=='zero' else COEFFICIENTS[variant]
    score,origins=held;score=np.asarray(score,float);origins=np.asarray(origins)
    if score.shape!=np.shape(panel['eligible']) or score.shape[1]!=len(dates) or origins.shape!=(len(dates),):
        raise ValueError('Aligned forecast, origin and market dates required')
    out=np.full_like(score,np.nan);receipts=[]
    for d in range(1,len(dates)):
        valid=np.asarray(panel['eligible'][:,d-1],bool)&np.isfinite(score[:,d-1])
        ids=np.flatnonzero(valid);base=np.full(score.shape[0],np.nan)
        if len(ids):
            assert 0<=origins[d-1]<=d-1
            base[ids]=(ranks(score[ids,d-1])+1)/len(ids)
        opening=np.asarray(panel['o'][:,d],float)
        previous=np.asarray(panel['close'][:,d-1],float)
        vol=np.asarray(panel['vol20'][:,d-1],float)
        usable=valid&np.isfinite(opening)&(opening>0)&np.isfinite(previous)&(previous>0)&np.isfinite(vol)&(vol>=0)
        gap_ids=np.flatnonzero(usable);applied=len(gap_ids)>=20
        if applied:
            gap=np.log(opening[gap_ids]/previous[gap_ids])/np.maximum(vol[gap_ids],1e-4)
            base[gap_ids]+=coefficient*((ranks(gap)+1)/len(gap_ids)-.5)
        out[:,d]=base
        receipts.append(dict(decision_date=dates[d],forecast_asof_date=dates[d-1],
            forecast_origin_date=dates[int(origins[d-1])] if len(ids) else None,
            last_input_time='09:00:00',execution_not_before='09:05:00',eligible_forecasts=len(ids),
            usable_gaps=len(gap_ids),overlay_available=applied))
    return out,dict(variant=variant,coefficient=coefficient,alignment='decision_date',receipts=receipts,
        inputs=['prior_eligible','held_prior_forecast','current_open','previous_close','previous_vol20'],
        future_execution_data_used=False,raw_gap_without_inferred_split_filter=True,
        missingness='Zero overlay for unavailable gap; fewer than20 gaps means zero overlay for everyone.')


def identity(model,variant,policy):
    return 'opening_'+variant+'_'+account_id(model,policy)


def hypotheses(cases):
    rows=[]
    for case in cases:
        for member in MEMBERS:
            for variant in VARIANTS:
                for policy in policy_grid():
                    model=f'flt_{case}_trained_{member}'
                    key=identity(model,variant,policy)
                    untrained=identity(f'flt_{case}_untrained_{member}',variant,policy)
                    rows.extend([(key,'cash',None),(key,'matching_untrained_opening',untrained),
                        (key,'same_opening_nonimage',identity('online_nonimage',variant,policy)),
                        (key,'unchanged_original',account_id(model,policy)),(untrained,'cash',None)])
    rows.extend((identity('online_nonimage',v,p),'cash',None) for v in VARIANTS for p in policy_grid())
    assert len(rows)==len({(k,c) for k,c,_ in rows})
    return rows
