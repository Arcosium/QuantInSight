"""Offline longer-hold, low-turnover spot study. No credentials or order API.

Frozen exploratory alternative to per-burst trading: aggregate causal refill
states and retain inventory until an hour-scale forecast expires or reverses.
Chronological 21/7/1 train/calibration/test; never claim reused history proves
future profitability. Both families pay the same 10bp/side fee.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from quant.impact_alpha_backtest import sha
from quant.impact_alpha_rolling import windows
from quant.impact_alpha_stats import add_net_columns, summarize_trades
from quant.impact_backfill import load_day
from quant.impact_longitudinal import clean_json

HOLDS = (60,240)
QUANTILES = (.80,.95)
FEE = 10.
CAPITAL = 100000.
TRADE_COLUMNS = ['day','decision_ms','entry_ms','exit_ms','entry_price','exit_price',
                 'quantity','entry_quote_ms','exit_quote_ms','hold_minutes',
                 'exit_mode','exit_reason','exit_wait_minutes','entry_score','threshold']
_DATA = None
_OUTPUT = None


def execution_labels(frame):
    """Prices one minute after decision; keep rows with unavailable outcomes."""
    out=frame.sort_values('decision_ms').reset_index(drop=True).copy()
    times=out.decision_ms.to_numpy(np.int64)
    if len(out)>1 and not np.all(np.diff(times)==60000):
        raise ValueError('Retain every calendar minute, including invalid rows')
    # Same known entry cutoff for both holds. Reserve5m for a delayed exit.
    minute=(times%86400000)//60000
    out['entry_allowed']=minute<=1440-1-max(HOLDS)-1-5
    entry=out.buy_vwap.shift(-1)
    ev=out.buy_valid.shift(-1,fill_value=False).astype(bool)
    for hold in HOLDS:
        exit_price=out.sell_vwap.shift(-1-hold)
        xv=out.sell_valid.shift(-1-hold,fill_value=False).astype(bool)
        mask=ev & xv & out.entry_allowed
        out[f'label_{hold}']=np.where(mask,(exit_price/entry-1)*10000,np.nan)
        out[f'label_exit_ms_{hold}']=out.decision_ms+(hold+1)*60000
    return out


def replay(frame,scores,threshold,hold,exit_mode):
    """Replay sequential signals; exits respond only to later-known scores.

    Entry/exit orders are priced at the next minute's visible .1ETH VWAP.
    Unpriced entries are rejected at execution time. An unpriced exit remains
    pending for at most5m, then fails the entire research run, never disappears.
    """
    if exit_mode not in ('fixed','reverse15'):raise ValueError('Unknown exit mode')
    if hold not in HOLDS:raise ValueError('Unsupported holding time')
    scores=np.asarray(scores,float)
    if len(frame)!=len(scores):raise ValueError('Score alignment mismatch')
    if frame.empty:return pd.DataFrame(columns=TRADE_COLUMNS),{'rejected_entries':0}
    ts=frame.decision_ms.to_numpy(np.int64)
    if len(ts)>1 and not np.all(np.diff(ts)==60000):raise ValueError('Missing calendar minute')
    valid=frame.state_valid.to_numpy(bool)
    buy_ok=frame.buy_valid.to_numpy(bool);sell_ok=frame.sell_valid.to_numpy(bool)
    schedule=frame.entry_allowed.to_numpy(bool)
    buy=frame.buy_vwap.to_numpy(float);sell=frame.sell_vwap.to_numpy(float)
    qt=frame.quote_ms.to_numpy(float)
    rows=[];rejected=0;i=0
    while i<len(frame):
        if not(valid[i] and schedule[i] and np.isfinite(scores[i]) and scores[i]>=threshold):
            i+=1;continue
        entry=i+1
        if entry>=len(frame):raise ValueError('Selected entry beyond frame')
        if not buy_ok[entry]:rejected+=1;i+=1;continue
        due=entry+hold;reason='expiry'
        if due>=len(frame):raise ValueError('Selected exit beyond frame')
        if exit_mode=='reverse15':
            candidates=np.flatnonzero(valid[entry+15:due] & np.isfinite(scores[entry+15:due])
                                      & (scores[entry+15:due]<0))
            if len(candidates):due=entry+15+int(candidates[0])+1;reason='forecast_reversal'
        exits=np.flatnonzero(sell_ok[due:min(due+6,len(frame))])
        if not len(exits):raise ValueError(f'Unpriced selected exit {ts[due]}')
        exit_index=due+int(exits[0])
        if ts[i]//86400000 != ts[exit_index]//86400000:raise ValueError('Position crossed day boundary')
        rows.append({'day':str(frame.iloc[i].day),'decision_ms':int(ts[i]),
                     'entry_ms':int(ts[entry]),'exit_ms':int(ts[exit_index]),
                     'entry_price':float(buy[entry]),'exit_price':float(sell[exit_index]),
                     'quantity':.1,'entry_quote_ms':float(qt[entry]),'exit_quote_ms':float(qt[exit_index]),
                     'hold_minutes':hold,'exit_mode':exit_mode,'exit_reason':reason,
                     'exit_wait_minutes':exit_index-due,'entry_score':float(scores[i]),
                     'threshold':float(threshold)})
        i=exit_index
    return pd.DataFrame(rows,columns=TRADE_COLUMNS),{'rejected_entries':rejected}


def choose(candidates):
    # Calibration is a selection sample, not a proof of profitability.
    eligible=[c for c in candidates if c['calibration']['n']>=14
              and c['calibration']['trading_days']>=4 and c['calibration']['net_pnl']>0]
    return max(eligible,key=lambda c:(c['calibration']['net_pnl'],-c['hold'],c['quantile'],c['exit_mode'])) if eligible else None


def build_cached_day(job):
    from quant.impact_alpha_state_features import build_minute_quotes
    cache,output=map(Path,job)
    data=load_day(cache);day=data['manifest']['day_utc']
    frame=build_minute_quotes(data,day)
    destination=output/'minutes'/day;destination.mkdir(parents=True,exist_ok=False)
    frame.to_parquet(destination/'quotes.parquet',index=False)
    (destination/'source.json').write_text(json.dumps({'cache_manifest_sha256':sha(cache/'manifest.json')}))
    return day,int(frame.quote_valid.sum())


def initialize(dataset,output):
    global _DATA,_OUTPUT
    _DATA=pd.read_parquet(dataset).sort_values('decision_ms').reset_index(drop=True)
    _OUTPUT=Path(output)


def fit_fold(spec):
    from quant.impact_alpha_state_features import CONTEXT_COLUMNS,MICRO_COLUMNS
    train=_DATA[_DATA.day.isin(spec['train'])]
    cal=_DATA[_DATA.day.isin(spec['calibration'])]
    test=_DATA[_DATA.day==spec['test']]
    cal_start=int(pd.Timestamp(spec['calibration'][0],tz='UTC').timestamp()*1000)
    families={'context':list(CONTEXT_COLUMNS),'full':list(CONTEXT_COLUMNS)+list(MICRO_COLUMNS)}
    models=[];predictions={};all_candidates={};selections={};trades=[]
    for family,features in families.items():
        for hold in HOLDS:
            fitting=train[train.state_valid & train[f'label_{hold}'].notna()
                          & (train[f'label_exit_ms_{hold}']<cal_start)]
            if len(fitting)<1000:raise ValueError('Insufficient past training data')
            x=fitting[features].to_numpy(float);y=fitting[f'label_{hold}'].to_numpy(float)
            lo,hi=np.quantile(x,[.01,.99],axis=0);yl,yh=np.quantile(y,[.01,.99])
            scaler=StandardScaler();x=scaler.fit_transform(np.clip(x,lo,hi))
            with threadpool_limits(limits=2):
                model=Ridge(alpha=1000.).fit(x,np.clip(y,yl,yh))
            def predict(part):
                values=model.predict(scaler.transform(np.clip(part[features].to_numpy(float),lo,hi)))
                return np.where(part.state_valid,values,np.nan)
            cp,tp=predict(cal),predict(test)
            predictions[(family,hold)]=(cp,tp)
            usable=cp[cal.state_valid.to_numpy(bool)&cal.entry_allowed.to_numpy(bool)]
            cutoffs={q:float(max(np.quantile(usable,q),2*FEE+1)) for q in QUANTILES}
            models.append({'family':family,'hold':hold,'features':features,'n_train':len(fitting),
                           'target_clip':[float(yl),float(yh)],'feature_clip_low':lo.tolist(),
                           'feature_clip_high':hi.tolist(),'cutoffs':cutoffs,
                           'scaled_coefficients':model.coef_.tolist(),'intercept':float(model.intercept_)})
        candidates=[]
        for hold in HOLDS:
            meta=next(m for m in models if m['family']==family and m['hold']==hold)
            for q,threshold in meta['cutoffs'].items():
                for mode in ('fixed','reverse15'):
                    rows,_=replay(cal,predictions[(family,hold)][0],threshold,hold,mode)
                    stat=summarize_trades(rows,spec['calibration'],fee_bps=FEE,initial_capital=CAPITAL,repetitions=500)
                    candidates.append({'hold':hold,'quantile':q,'threshold':threshold,'exit_mode':mode,'calibration':stat})
        selected=choose(candidates);all_candidates[family]=candidates
        if selected is None:
            rows=pd.DataFrame(columns=TRADE_COLUMNS)
            selections[family]={'action':'cash','reason':'No positive past-net candidate with14 trades on4 days'}
        else:
            rows,execution=replay(test,predictions[(family,selected['hold'])][1],selected['threshold'],selected['hold'],selected['exit_mode'])
            selections[family]={'action':'trade_if_signal','chosen':selected,'execution':execution}
        rows=add_net_columns(rows,fee_bps=FEE);rows['family']=family;trades.append(rows)
    dest=_OUTPUT/'folds'/spec['test'];dest.mkdir(parents=True,exist_ok=False)
    parts=[r for r in trades if len(r)]
    combined=pd.concat(parts,ignore_index=True) if parts else trades[0]
    combined.to_csv(dest/'trades.csv.gz',index=False)
    score=test[['day','decision_ms','state_valid','entry_allowed','label_60','label_240']].copy()
    for (family,hold),(_,values) in predictions.items():score[f'pred_{family}_{hold}']=values
    score.to_parquet(dest/'predictions.parquet',index=False)
    (dest/'selection.json').write_text(json.dumps(clean_json({'window':spec,'models':models,
                      'candidates':all_candidates,'selections':selections}),indent=2))
    return spec['test'],{k:v['action'] for k,v in selections.items()},len(combined)


def block_interval(values,seed=928,repetitions=5000):
    values=np.asarray(values,float);n=len(values)
    if n<2:return None
    starts=np.random.default_rng(seed).integers(0,n,size=(repetitions,int(np.ceil(n/7))))
    indices=((starts[:,:,None]+np.arange(7)[None,None,:])%n).reshape(repetitions,-1)[:,:n]
    return np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()


def summarize(output):
    folds=sorted((output/'folds').iterdir());days=[p.name for p in folds]
    parts=[pd.read_csv(p/'trades.csv.gz') for p in folds]
    nonempty=[p for p in parts if len(p)]
    all_trades=pd.concat(nonempty,ignore_index=True) if nonempty else parts[0]
    data=pd.read_parquet(output/'dataset.parquet')
    summary={'evaluation_days':days,'fee_per_side_bps':FEE,'actual_orders':0,'families':{},
             'status':'exploratory_reused_history','equity_basis':'valid-minute liquidation equity; intraminute risk not measured'}
    daily={}
    for family in ('full','context'):
        rows=all_trades[all_trades.family==family].sort_values('entry_ms')
        rows.to_csv(output/f'{family}_trades.csv',index=False)
        stat=summarize_trades(rows,days,fee_bps=FEE,initial_capital=CAPITAL)
        daily[family]=rows.groupby('day').net_pnl.sum().reindex(days,fill_value=0)
        stat['daily_pnl_7day_block_ci95']=block_interval(daily[family]) if len(rows) else None
        stat['first_half_net_pnl']=float(daily[family].iloc[:len(days)//2].sum())
        stat['second_half_net_pnl']=float(daily[family].iloc[len(days)//2:].sum())
        stat['extra_1bp_net_pnl']=summarize_trades(rows,days,fee_bps=FEE,extra_slippage_bps=1,initial_capital=CAPITAL)['net_pnl']
        stat['exposure_hours']=float((rows.exit_ms-rows.entry_ms).sum()/3600000)
        stat['cash_policy_days']=sum(json.loads((p/'selection.json').read_text())['selections'][family]['action']=='cash' for p in folds)
        cash=CAPITAL;peak=CAPITAL;max_dd=0.
        for r in rows.itertuples():
            cost=r.quantity*r.entry_price*(1+FEE/10000)
            if cash<cost:raise ValueError('Capital exhausted')
            marks=data[(data.decision_ms>=r.entry_ms)&(data.decision_ms<=r.exit_ms)&data.sell_valid].sell_vwap.to_numpy()
            equity=cash-cost+r.quantity*marks*(1-FEE/10000)
            running=np.maximum.accumulate(np.r_[peak,equity])[1:]
            if len(equity):max_dd=max(max_dd,float(np.max((running-equity)/running)));peak=max(peak,float(np.max(equity)))
            cash+=r.net_pnl;peak=max(peak,cash)
        stat['valid_minute_max_drawdown_pct']=max_dd*100
        summary['families'][family]=stat
    difference=daily['full']-daily['context']
    improvement_ci=block_interval(difference)
    s=summary['families']['full']
    checks={'positive_net_pnl':s['net_pnl']>0,'at_least100_trades':s['n']>=100,
            'at_least20_trading_days':s['trading_days']>=20,
            'positive_net_daily_lower_ci':s['daily_pnl_7day_block_ci95'] is not None and s['daily_pnl_7day_block_ci95'][0]>0,
            'increment_over_context_lower_ci':improvement_ci is not None and improvement_ci[0]>0,
            'both_test_halves_positive':min(s['first_half_net_pnl'],s['second_half_net_pnl'])>0,
            'extra_slippage_survives':s['extra_1bp_net_pnl']>0}
    summary['increment_over_context']={'total_pnl':float(difference.sum()),'mean_daily_pnl_ci95':improvement_ci}
    summary['predefined_checks']=checks;summary['internal_pass']=all(checks.values())
    pd.DataFrame(daily).assign(full_minus_context=difference).to_csv(output/'daily_pnl.csv')
    (output/'results.json').write_text(json.dumps(clean_json(summary),indent=2))
    return summary


def main():
    from quant.impact_alpha_state_features import add_state_features,CONTEXT_COLUMNS,MICRO_COLUMNS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume-models',action='store_true')
    args=parser.parse_args();out=args.output
    if not args.resume_models:
        out.mkdir(parents=True,exist_ok=False)
        protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'actual_orders':0,
                  'code_sha256':sha(__file__),'features_code_sha256':sha(Path(__file__).with_name('impact_alpha_state_features.py')),
                  'events_sha256':sha(args.events),'window':'21train/7cal/1test, step1day',
                  'holds_minutes':HOLDS,'score_quantiles':QUANTILES,'exit_modes':['fixed','reverse15'],
                  'threshold':'max(past-calibration-score-quantile,21bp)',
                  'selection':'highest past7day netUSDT; minimum14 trades on4days; positive net; else cash',
                  'model':'Ridge alpha1000, training-only scaler and1/99 winsorization, no hyperparameter search',
                  'fee_per_side_bps':FEE,'quantity_eth':.1,'capital_usdt':CAPITAL,'execution_delay_seconds':60,
                  'exit_missing':'wait up to5 calendar minutes then fail; entries reject unavailable execution quotes',
                  'common_entry_cutoff_utc':'19:53','future_evaluation_success':['net positive','n>=100','tradingdays>=20',
                    '7day-block mean daily net lower>0','7day-block increment over context lower>0',
                    'both test halves positive','extra1bp/side net positive'],
                  'limitation':'Exploratory second design on already inspected history; no claim of untouched validation'}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        caches=sorted(p for p in args.cache_root.iterdir() if (p/'manifest.json').exists())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for day,n in pool.map(build_cached_day,[(str(p),str(out)) for p in caches]):
                print(json.dumps({'phase':'minutes','day':day,'valid':n}),flush=True)
        quotes=pd.concat([pd.read_parquet(p) for p in sorted((out/'minutes').glob('*/quotes.parquet'))],ignore_index=True)
        events=pd.read_parquet(args.events)
        states=add_state_features(quotes,events)
        columns=list(CONTEXT_COLUMNS)+list(MICRO_COLUMNS)
        if not np.isfinite(states[columns]).all().all():raise ValueError('Nonfinite model features')
        states=execution_labels(states)
        states.to_parquet(out/'dataset.parquet',index=False)
        print(json.dumps({'phase':'features','rows':len(states),'state_valid':int(states.state_valid.sum())}),flush=True)
    days=sorted(pd.read_parquet(out/'dataset.parquet',columns=['day']).day.unique())
    if days!=pd.date_range(days[0],days[-1]).strftime('%Y-%m-%d').tolist():raise ValueError('Nonconsecutive input days')
    specs=windows(days)
    todo=[s for s in specs if not (out/'folds'/s['test']/'selection.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(str(out/'dataset.parquet'),str(out))) as pool:
        for day,selection,n in pool.map(fit_fold,todo):
            print(json.dumps({'phase':'fold','day':day,'selection':selection,'n':n}),flush=True)
    summary=summarize(out)
    print(json.dumps({'phase':'complete','internal_pass':summary['internal_pass'],'output':str(out)}),flush=True)


if __name__=='__main__':main()
