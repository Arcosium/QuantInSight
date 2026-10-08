"""Offline walk-forward alpha research with train/calibration/test chronology.

Model and entry thresholds are selected using past observations only. A cash
policy is an explicit outcome, never evidence of profitable alpha. No orders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from quant.impact_alpha_backtest import sha
from quant.impact_alpha_stats import summarize_trades, add_net_columns
from quant.impact_backfill import load_day
from quant.impact_longitudinal import clean_json

HORIZONS = (30, 300, 1800)
QUANTILES = (.90, .97, .99)
POLICIES = {
    'full_cost10': ('full', 10, True),
    'context_cost10': ('context', 10, True),
    'full_cost1': ('full', 1, True),
    'full_rank97': ('full', 10, False),
    'context_rank97': ('context', 10, False),
}
CAPITAL = 100000
TRADE_COLUMNS = ['day','event_ms','decision_ms','entry_ms','exit_ms','entry_price',
                 'exit_price','quantity','entry_quote_ms','exit_quote_ms','hold_seconds',
                 'predicted_gross_bps','threshold_bps']
_DATA = None
_OUTPUT = None
_FEATURES = None


def windows(days, training_days=21, calibration_days=7):
    ordered = sorted(set(str(d) for d in days))
    offset = training_days+calibration_days
    return [{'train':ordered[i-offset:i-calibration_days],
             'calibration':ordered[i-calibration_days:i], 'test':day}
            for i,day in enumerate(ordered) if i >= offset]


def replay(frame, predictions, threshold, horizon):
    """Enter only after a past-only score; missing selected exits are fatal."""
    if len(frame) != len(predictions): raise ValueError('Prediction alignment mismatch')
    rows = []; free_at = -10**18
    audit = {'triggered':0,'occupied':0,'entry_invalid':0}
    for row, score in zip(frame.itertuples(), predictions):
        if not np.isfinite(score) or score < threshold: continue
        audit['triggered'] += 1
        if row.decision_ms < free_at:
            audit['occupied'] += 1; continue
        if not row.entry_valid:
            audit['entry_invalid'] += 1; continue
        if not getattr(row,f'exit_valid_{horizon}'):
            raise ValueError(f'Unpriced selected exit {row.day} {row.event_ms} {horizon}')
        exit_ms = int(getattr(row,f'exit_ms_{horizon}'))
        rows.append({'day':row.day,'event_ms':int(row.event_ms),'decision_ms':int(row.decision_ms),
                     'entry_ms':int(row.entry_ms),'exit_ms':exit_ms,'entry_price':row.entry_price,
                     'exit_price':getattr(row,f'exit_price_{horizon}'),'quantity':row.quantity,
                     'entry_quote_ms':int(row.entry_quote_ms),
                     'exit_quote_ms':int(getattr(row,f'exit_quote_ms_{horizon}')),
                     'hold_seconds':horizon,'predicted_gross_bps':float(score),
                     'threshold_bps':float(threshold)})
        free_at = exit_ms
    return pd.DataFrame(rows,columns=TRADE_COLUMNS),audit


def pick_candidate(candidates, gated=True):
    eligible = []
    for c in candidates:
        s = c['calibration']
        if s['n'] < 30 or s['trading_days'] < 4: continue
        if gated:
            ci = s['mean_net_bps_ci95']
            if ci is None or ci[0] <= 0: continue
            score = ci[0]
        else:
            score = s['mean_gross_bps']
        if score is not None and np.isfinite(score):
            eligible.append((score,-c['horizon'],c['quantile'],c))
    return max(eligible,key=lambda x:x[:3])[-1] if eligible else None


def build_cached_day(job):
    from quant.impact_alpha_features import build_day_frame
    cache, output = map(Path,job)
    data = load_day(cache)
    day = data['manifest']['day_utc']
    end_ms = int((pd.Timestamp(day,tz='UTC')+pd.Timedelta(days=1)).timestamp()*1000)
    frame,audit = build_day_frame(data,day,end_ms,threshold=1000.,horizons=HORIZONS)
    destination = output/'dataset'/day
    destination.mkdir(parents=True,exist_ok=False)
    frame.to_parquet(destination/'frame.parquet',index=False)
    audit['cache_manifest_sha256'] = sha(cache/'manifest.json')
    (destination/'audit.json').write_text(json.dumps(clean_json(audit),indent=2))
    return day,len(frame)


def initialize_worker(dataset, output, feature_columns):
    global _DATA,_OUTPUT,_FEATURES
    _DATA = pd.read_parquet(dataset).sort_values(['day','decision_ms']).reset_index(drop=True)
    _OUTPUT = Path(output)
    _FEATURES = tuple(feature_columns)


def context_columns(columns):
    allowed = ('f_sign','f_log_notional','f_log_quantity','f_dominance',
               'f_opposing_flow_share_1s','f_log_flow_quantity_1s',
               'f_return_10s_bps','f_return_60s_bps','f_volatility_10s_bps',
               'f_volatility_60s_bps','f_log_activity_10s','f_buy_fraction_10s',
               'f_utc_time_sin','f_utc_time_cos')
    return [name for name in columns if name in allowed]


def fit_fold(spec):
    train = _DATA[_DATA.day.isin(spec['train'])]
    cal = _DATA[_DATA.day.isin(spec['calibration'])]
    test = _DATA[_DATA.day == spec['test']]
    if train.empty or cal.empty: raise ValueError('Empty historical window')
    if max(spec['train']) >= min(spec['calibration']) or max(spec['calibration']) >= spec['test']:
        raise ValueError('Invalid chronological boundary')
    cal_start = int(pd.Timestamp(spec['calibration'][0],tz='UTC').timestamp()*1000)
    test_start = int(pd.Timestamp(spec['test'],tz='UTC').timestamp()*1000)
    cache = {}; model_meta = []
    feature_sets = {'full':list(_FEATURES),'context':context_columns(_FEATURES)}
    if not feature_sets['context'] or feature_sets['context']==feature_sets['full']:
        raise ValueError('Context ablation must exclude order-book and post-event impact features')
    with threadpool_limits(limits=2):
        for family,features in feature_sets.items():
            for horizon in HORIZONS:
                valid = train.entry_valid & train[f'exit_valid_{horizon}']
                fitting = train.loc[valid]
                if fitting[f'exit_ms_{horizon}'].max() >= cal_start:
                    raise ValueError('Training label crosses calibration boundary')
                cv = cal.loc[cal.entry_valid & cal[f'exit_valid_{horizon}']]
                if len(cv) and cv[f'exit_ms_{horizon}'].max() >= test_start:
                    raise ValueError('Calibration label crosses test boundary')
                y = fitting[f'gross_bps_{horizon}'].to_numpy(float)
                low,high = np.quantile(y,[.005,.995])
                model = HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,
                    learning_rate=.05,min_samples_leaf=200,l2_regularization=10.,
                    early_stopping=False,random_state=928)
                model.fit(fitting[features],np.clip(y,low,high))
                prediction_train = model.predict(fitting[features])
                prediction_cal = model.predict(cal[features])
                prediction_test = model.predict(test[features]) if len(test) else np.array([])
                cutoffs = {q:float(np.quantile(prediction_train,q)) for q in QUANTILES}
                cache[(family,horizon)] = (prediction_cal,prediction_test,cutoffs)
                model_meta.append({'family':family,'horizon':horizon,'n_train':len(fitting),
                    'features':features,'target_clip':[float(low),float(high)],'train_score_quantiles':cutoffs})
    selections = {}; all_candidates = {}; frames = []
    for policy,(family,fee,gated) in POLICIES.items():
        candidates = []
        for horizon in HORIZONS:
            pred_cal,_,cutoffs = cache[(family,horizon)]
            for q in (QUANTILES if gated else (.97,)):
                # The required return covers BOTH fees plus a 1 bp uncertainty buffer.
                threshold = max(cutoffs[q],2*fee+1 if gated else 0.)
                trades,_ = replay(cal,pred_cal,threshold,horizon)
                stat = summarize_trades(trades,spec['calibration'],fee_bps=fee,
                                        initial_capital=CAPITAL,repetitions=500)
                candidates.append({'family':family,'horizon':horizon,'quantile':q,
                                   'threshold':threshold,'calibration':stat})
        chosen = pick_candidate(candidates,gated=gated)
        all_candidates[policy] = candidates
        if chosen is None:
            rows = pd.DataFrame(columns=TRADE_COLUMNS); audit = {}
            selections[policy] = {'action':'cash','reason':'No past calibration candidate passed the predefined criteria'}
        else:
            pred = cache[(family,chosen['horizon'])][1]
            rows,audit = replay(test,pred,chosen['threshold'],chosen['horizon'])
            selections[policy] = {'action':'trade_if_signal','chosen':chosen,'test_execution_audit':audit}
        rows['policy'] = policy
        frames.append(add_net_columns(rows,fee_bps=fee))
    dest = _OUTPUT/'folds'/spec['test'];dest.mkdir(parents=True,exist_ok=False)
    nonempty = [f for f in frames if len(f)]
    trades = pd.concat(nonempty,ignore_index=True) if nonempty else frames[0].iloc[:0]
    trades.to_csv(dest/'trades.csv.gz',index=False)
    score_frame = test[['day','event_ms','decision_ms','entry_valid',
                       *[f'gross_bps_{h}' for h in HORIZONS]]].copy()
    for (family,horizon),(_,pred,_) in cache.items():score_frame[f'pred_{family}_{horizon}']=pred
    score_frame.to_parquet(dest/'predictions.parquet',index=False)
    meta = {'window':spec,'models':model_meta,'selections':selections,'calibration_candidates':all_candidates}
    (dest/'selection.json').write_text(json.dumps(clean_json(meta),ensure_ascii=False,indent=2))
    return spec['test'], {key:val['action'] for key,val in selections.items()},len(trades)


def moving_block_ci(trades, days, length=7, repetitions=2000):
    """Circular blocks of calendar days preserve short serial dependence."""
    if trades.empty or len(days)<2:return None
    table=trades.groupby('day').agg(total=('net_bps','sum'),count=('net_bps','size')).reindex(days,fill_value=0)
    rng=np.random.default_rng(928)
    starts=rng.integers(0,len(days),size=(repetitions,int(np.ceil(len(days)/length))))
    sampled=(starts[:,:,None]+np.arange(length)[None,None,:])%len(days)
    sampled=sampled.reshape(repetitions,-1)[:,:len(days)]
    totals=table.total.to_numpy()[sampled].sum(axis=1)
    counts=table['count'].to_numpy()[sampled].sum(axis=1)
    valid=counts>0
    return np.quantile(totals[valid]/counts[valid],[.025,.975]).tolist() if valid.any() else None


def summarize_run(output, live_day=None):
    output=Path(output)
    folds=sorted((output/'folds').iterdir())
    historical=[p for p in folds if p.name!=live_day]
    all_frames=[pd.read_csv(p/'trades.csv.gz') for p in historical]
    nonempty=[f for f in all_frames if len(f)]
    all_trades=pd.concat(nonempty,ignore_index=True) if nonempty else all_frames[0]
    days=[p.name for p in historical]
    metas=[json.loads((p/'selection.json').read_text()) for p in historical]
    result={'evaluation_days':days,'policies':{},'actual_orders':0,
            'inference':'Exploratory re-used data; not an untouched prospective sample. No guarantee of profitable alpha.',
            'selection_trials_per_fold':{'full_cost10':9,'context_cost10':9,'full_cost1':9,'full_rank97':3,'context_rank97':3},
            'drawdown_basis':'Closed-trade realized equity only; intratrade risk excluded.'}
    for policy,(_,fee,gated) in POLICIES.items():
        rows=all_trades[all_trades.policy==policy].sort_values('entry_ms').copy()
        stat=summarize_trades(rows,days,fee_bps=fee,initial_capital=CAPITAL)
        stat['moving_7day_block_mean_net_ci95']=moving_block_ci(rows,days)
        cash_days=sum(meta['selections'][policy]['action']=='cash' for meta in metas)
        stat['cash_policy_days']=cash_days
        stat['cost_gate']=gated
        if len(rows):
            cash=CAPITAL+np.r_[0,rows.net_pnl.to_numpy()[:-1].cumsum()]
            if np.any(cash<rows.entry_price.to_numpy()*.1*(1+fee/1e4)):
                raise ValueError('Capital exhausted')
            if np.any(rows.entry_ms.to_numpy()[1:]<rows.exit_ms.to_numpy()[:-1]):
                raise ValueError('Overlapping portfolio positions')
        rows.to_csv(output/f'{policy}_trades.csv',index=False)
        result['policies'][policy]={'historical':stat,
            'extra_slippage_1bp':summarize_trades(rows,days,fee_bps=fee,extra_slippage_bps=1,initial_capital=CAPITAL)}
        if live_day:
            live=pd.read_csv(output/'folds'/live_day/'trades.csv.gz')
            live=live[live.policy==policy]
            result['policies'][policy]['live']=summarize_trades(live,[live_day],fee_bps=fee,initial_capital=CAPITAL)
    (output/'results.json').write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2))
    return result


def main():
    from quant.impact_alpha_features import FEATURE_COLUMNS,build_day_frame
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live-input',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume-models',action='store_true')
    args=parser.parse_args()
    output=args.output
    if not args.resume_models:
        output.mkdir(parents=True,exist_ok=False)
        protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'actual_orders':0,
            'training_days':21,'calibration_days':7,'test_days':1,'step_days':1,
            'fixed_notional_threshold_usdt':1000.,'horizons_seconds':HORIZONS,
            'prediction_quantiles':QUANTILES,'minimum_calibration_trades':30,'minimum_calibration_trading_days':4,
            'gate':'Predicted gross >=2*fee+1bp; past calibration day-bootstrap net CI lower >0',
            'research_policy':'rank97 probes have no profitability gate and are not deployment candidates',
            'quantity_eth':.1,'capital_usdt':CAPITAL,'entry_latency_ms':100,
            'code_sha256':sha(__file__),'feature_code_sha256':sha(Path(__file__).with_name('impact_alpha_features.py')),
            'model':'HistGradientBoosting,80 iterations,7 leaves,minleaf200,l2=10,learning_rate=.05,no random early stopping',
            'fee_source':'https://www.bybit.com/en/help-center/article/Trading-Fee-Structure',
            'limits':'All historical periods were already inspected in earlier research; this is exploratory walk-forward reuse.'}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        caches=sorted(p for p in args.cache_root.iterdir() if (p/'manifest.json').exists())
        with ProcessPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
            for day,n in pool.map(build_cached_day,[(str(p),str(output)) for p in caches]):
                print(json.dumps({'phase':'features','day':day,'n':n}),flush=True)
        parts=[pd.read_parquet(p) for p in sorted((output/'dataset').glob('*/frame.parquet'))]
        if args.live_input:
            from quant.impact_live_study import read_completed_source
            data,manifest,source=read_completed_source(args.live_input)
            end=int(datetime.fromisoformat(manifest['ended_utc']).timestamp()*1000)
            day=datetime.fromisoformat(manifest['started_utc']).strftime('%Y-%m-%d')
            frame,audit=build_day_frame(data,day,end,threshold=1000.,horizons=HORIZONS)
            dest=output/'dataset'/day;dest.mkdir(parents=True,exist_ok=False)
            frame.to_parquet(dest/'frame.parquet',index=False)
            (dest/'audit.json').write_text(json.dumps(clean_json(audit),indent=2))
            (output/'live_source.json').write_text(json.dumps(source,indent=2))
            parts.append(frame)
        pd.concat(parts,ignore_index=True).to_parquet(output/'dataset.parquet',index=False)
    dataset=pd.read_parquet(output/'dataset.parquet',columns=['day'])
    days=sorted(dataset.day.unique())
    specs=windows(days)
    todo=[spec for spec in specs if not (output/'folds'/spec['test']/'selection.json').exists()]
    with ProcessPoolExecutor(max_workers=max(1,min(args.workers,4)),initializer=initialize_worker,
                             initargs=(str(output/'dataset.parquet'),str(output),FEATURE_COLUMNS)) as pool:
        for day,selections,n in pool.map(fit_fold,todo):
            print(json.dumps({'phase':'fold','day':day,'selection':selections,'n':n}),flush=True)
    live_day=days[-1] if (output/'live_source.json').exists() else None
    summarize_run(output,live_day)
    print(json.dumps({'phase':'complete','output':str(output)}),flush=True)


if __name__=='__main__':main()
