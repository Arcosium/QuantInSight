"""Four fixed score attribution controls; registration is stored separately.

Remove same-date score dependence on known size, volatility and sector features.
This projects existing forecasts, never future returns, and creates no shorts.
"""
import numpy as np
from quant.timefolio_heatmap_transfer_diagnostics import ranks
from quant.timefolio_heatmap_fleet_accounts import policy_grid, account_id

VARIANTS=['remove_cap','remove_vol','remove_cap_vol','remove_cap_vol_sector']
MEMBERS=['seed17','seed29','seed43','ensemble']


def residual(values,design):
    values=np.asarray(values,float);design=np.asarray(design,float)
    if values.ndim!=1 or design.ndim!=2 or len(values)!=len(design):
        raise ValueError('Matching current cross-sectional arrays required')
    if not np.isfinite(values).all() or not np.isfinite(design).all():
        raise ValueError('Finite current observations required')
    coef,_,rank,_=np.linalg.lstsq(design,values,rcond=1e-10)
    out=values-design@coef
    error=float(np.max(np.abs(design.T@out))) if design.shape[1] else 0.
    assert error<1e-8
    return out,dict(coefficients=coef.tolist(),design_rank=int(rank),orthogonality_error=error)


def transform(score,panel,dates,variant):
    if variant not in VARIANTS:raise ValueError('Unknown factor projection')
    score=np.asarray(score,float);eligible=np.asarray(panel['eligible'],bool)
    if score.shape!=eligible.shape or score.shape[1]!=len(dates):raise ValueError('Aligned forecast axes required')
    cap='cap' in variant;vol='vol' in variant;sector='sector' in variant
    result=np.full_like(score,np.nan);receipts=[]
    for day,date in enumerate(dates):
        valid=eligible[:,day]&np.isfinite(score[:,day])
        if cap:valid &= np.isfinite(panel['market_cap'][:,day])&(panel['market_cap'][:,day]>0)
        if vol:valid &= np.isfinite(panel['vol20'][:,day])&(panel['vol20'][:,day]>=0)
        ids=np.flatnonzero(valid)
        if len(ids)<20:
            receipts.append(dict(date=date,observations=len(ids),projected=False));continue
        y=(ranks(score[ids,day])+1)/len(ids)
        columns=[np.ones(len(ids))];names=['intercept']
        if cap:
            columns.append((ranks(panel['market_cap'][ids,day])+1)/len(ids));names.append('size_rank')
        if vol:
            columns.append((ranks(panel['vol20'][ids,day])+1)/len(ids));names.append('volatility_rank')
        if sector:
            categories=np.asarray(panel['sector'])[ids]
            for code in np.unique(categories)[1:]:
                columns.append((categories==code).astype(float));names.append('sector_'+str(int(code)))
        x=np.column_stack(columns);value,proof=residual(y,x)
        result[ids,day]=value+y.mean()
        receipts.append(dict(date=date,observations=len(ids),projected=True,columns=names,**proof))
    return result,None,dict(variant=variant,uses_future_returns=False,uses_returns_as_fit_target=False,
        fit_target='Current forecast percentile rank',receipts=receipts,
        missingness='Current missing required factor values excluded; no future outcome availability filter.',
        interpretation='Current factor-score residual; no independent alpha claim.')


def identity(model,variant,policy):
    return 'factor_'+variant+'_'+account_id(model,policy)


def hypotheses(cases):
    rows=[]
    for case in cases:
        for member in MEMBERS:
            for variant in VARIANTS:
                for policy in policy_grid():
                    model=f'flt_{case}_trained_{member}'
                    key=identity(model,variant,policy)
                    untrained=identity(f'flt_{case}_untrained_{member}',variant,policy)
                    rows.extend([(key,'cash',None),(key,'matching_untrained_factor',untrained),
                        (key,'same_factor_nonimage',identity('online_nonimage',variant,policy)),
                        (key,'unchanged_original',account_id(model,policy)),(untrained,'cash',None)])
    rows.extend((identity('online_nonimage',v,p),'cash',None) for v in VARIANTS for p in policy_grid())
    assert len(rows)==len({(k,c) for k,c,_ in rows})
    return rows
