"""Local causal model training and Timefolio long-only account evaluator.

Reads the frozen ArcTrade study. Never imports broker/web modules or writes inputs.
All mutations are process-local; model outputs are cached once by model identity.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from .config import ARC, STUDY, RUNS
from .genetics import model_fingerprint, normalize


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def legacy():
    sys.path.insert(0,str(ARC))
    sys.path.insert(0,str(STUDY))
    import numpy as np
    from scipy.stats import rankdata
    import fixed_target_linear31_inputs as inputs
    import fixed_target_linear31_numpy as linear
    import additive_quadratic62_numpy as quadratic
    from run_tree_pilot import ACCOUNT,KIS_CACHE
    return np,rankdata,inputs,linear,quadratic,ACCOUNT,KIS_CACHE


def version():
    names=['fixed_target_linear31_inputs.py','fixed_target_linear31_numpy.py','additive_quadratic62_numpy.py',
           'replay_sizing100_stock20_cal5_u100_tiny_lambda.py','cached_dynamic_sector_overlay.py']
    receipts=[STUDY/n for n in names]+[Path(__file__)]
    return hashlib.sha256(''.join(sha(p) for p in receipts).encode()).hexdigest()


def model_cache(g,progress=lambda *_:None):
    g=normalize(g)
    np,rankdata,inputs,linear,quadratic,ACCOUNT,KIS_CACHE=legacy()
    v=version()+sha(ACCOUNT/'receipt.json')+sha(KIS_CACHE/'receipt.json')
    identity=model_fingerprint(g,v)
    root=RUNS/'models'/identity
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest_path=root/'manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            assert manifest['model_id']==identity and sha(manifest['scores_path'])==manifest['scores_sha256']
            progress('model_cache_hit',dict(model_id=identity))
            return manifest
        attempt=root/('build_'+str(time.time_ns()))
        attempt.mkdir()
        proofs=[]
        with np.load(ACCOUNT/'panel.npz') as z:eligible=z['eligible'].astype(bool)
        dates=np.load(KIS_CACHE/'dates.npy',mmap_mode='r')
        codes=np.load(KIS_CACHE/'codes.npy',mmap_mode='r')
        if g['family']=='nn_consensus':
            from frozen_downside33_tail_scores import load_scores
            stock=load_scores()
            a,b=stock['masked']['original31'],stock['masked']['added33']
            blend=np.full(a.shape,np.nan,np.float64)
            for day in range(len(dates)):
                ids=np.flatnonzero(np.isfinite(a[:,day]))
                assert np.array_equal(ids,np.flatnonzero(np.isfinite(b[:,day])))
                if len(ids):blend[ids,day]=np.minimum((rankdata(a[ids,day])-.5)/len(ids),(rankdata(b[ids,day])-.5)/len(ids))
            proofs=stock['model_proofs']
            provenance='Frozen audited original31/added33 seed17 consensus; no new neural training'
        elif (g['window'],g['target'],g['l2'],g['features'])==(0,'risk20',.001,'all'):
            if g['family']=='quadratic62':
                from frozen_additive_quadratic62_scores import load_scores
            else:
                from frozen_fixed_target_linear31_scores import load_scores
            stock=load_scores()
            masked=stock['masked']['original31']
            blend=np.full(masked.shape,np.nan,np.float64)
            for day in range(len(dates)):
                ids=np.flatnonzero(np.isfinite(masked[:,day]))
                if len(ids):blend[ids,day]=(rankdata(masked[ids,day])-.5)/len(ids)
            proofs=stock['model_proofs']
            provenance='Existing audited fixed-target model files referenced; no model copies'
        else:
            data=inputs.load()
            x=data['x']
            if g['features']=='price':
                x=x.copy()
                x[:,16:]=0
            y=data['utility']
            if g['target']!='risk20':
                y=np.full(len(data['ts']),np.nan,np.float32)
                valid=np.flatnonzero(np.isfinite(data['quality']))
                end=data['ends'][valid] if g['target']=='net20' else data['ts'][valid]+6
                raw=data['raw']
                simple=(raw[data['ks'][valid],end,0].astype(float)/raw[data['ks'][valid],data['ts'][valid]+1,0].astype(float)-1).astype(np.float32)
                cost=np.log((1.001*1.0005)/(.997*.9995))
                y[valid]=np.log1p(np.clip(simple,-.3,.3).astype(float))-cost
            assert np.array_equal(np.isfinite(y),np.isfinite(data['utility']))
            linear.L2=g['l2']  # Isolated child process; no source or shared service changed.
            learner=linear if g['family']=='linear31' else quadratic
            scores=np.full(data['raw'].shape[:2],np.nan,np.float32)
            refs=data['reference']['training_proofs']
            for i,ref in enumerate(refs):
                q=ref['quarter']
                start=int(np.searchsorted(dates,q+'01'))
                lower=max(20,start-22-(g['window']-1)) if g['window'] else 20
                train=np.flatnonzero(np.isfinite(data['quality'])&(data['ends']<start)&(data['ts']>=lower))
                assert data['ends'][train].max()<start
                weights=np.where(data['price'][train,12]<0,2.,1.)
                median,scale,z=learner.preprocessing(x[train])
                pairs=linear.fixed_pairs(y[train],data['ts'][train],weights)
                progress('training',dict(quarter=q,model_id=identity,rows=len(train),iteration=i+1,total=len(refs)))
                coef,diagnostic=learner.solve(z,pairs)
                model=learner.model(coef,median,scale,diagnostic)
                model.update(l2=g['l2'],quarter=q,max_label_end=int(data['ends'][train].max()),first_test_signal=start,
                             training_rows_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                             target_sha256=hashlib.sha256(y[train].tobytes()).hexdigest(),features=g['features'])
                path=attempt/(q+'_model.json')
                atomic_json(path,model)
                stop=int(np.searchsorted(dates,refs[i+1]['quarter']+'01')) if i+1<len(refs) else len(dates)
                test=np.flatnonzero((data['ts']>=start)&(data['ts']<stop))
                prediction=learner.predict(model,x[test])
                assert np.isfinite(prediction).all() and np.array_equal(prediction,learner.predict(json.loads(path.read_text()),x[test]))
                scores[data['ks'][test],data['ts'][test]]=prediction
                proofs.append(dict(quarter=q,model_reference=str(path),model_sha256=sha(path),
                                   train_rows=len(train),max_label_end=model['max_label_end'],first_test_signal=start,
                                   gradient_linf=diagnostic['gradient_linf'],target_sha256=model['target_sha256'],
                                   saved_prediction_exact=True,prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest()))
            from liquid_tiny_lambda_mask import mask_scores
            masked,membership=mask_scores(scores,data['raw'],codes,eligible,100)
            blend=np.full(masked.shape,np.nan,np.float64)
            for day in range(len(dates)):
                ids=np.flatnonzero(np.isfinite(masked[:,day]))
                if len(ids):blend[ids,day]=(rankdata(masked[ids,day])-.5)/len(ids)
            provenance='13 quarterly models trained locally with strictly matured labels'
        score_path=attempt/'scores.npy'
        np.save(score_path,blend)
        manifest=dict(model_id=identity,model_genes={k:g[k] for k in ['family','window','target','l2','features']},
                      scores_path=str(score_path),scores_sha256=sha(score_path),proofs=proofs,
                      provenance=provenance,version=v,shared_input_writes=0,new_input_copies=0)
        atomic_json(manifest_path,manifest)
        return manifest


def verify_book(case,panel,index):
    np,*_=legacy()
    from collections import defaultdict
    codes={c:i for i,c in enumerate(index['codes'])}
    dates={d:i for i,d in enumerate(index['dates'])}
    quantity=np.zeros(len(codes),np.int64)
    marks=np.zeros(len(codes))
    cash=1e9
    trades=defaultdict(list)
    for tr in case['trades']:trades[tr['date']].append(tr)
    for row in case['daily']:
        day=dates[row['date']]
        ratio=panel['split'][:,day]
        quantity=np.floor(quantity*ratio+1e-7).astype(np.int64)
        marks=np.divide(marks,ratio,out=marks.copy(),where=ratio>0)
        opening=panel['exec_price'][:,day].astype(float)
        good=np.isfinite(opening)&(opening>0)&(panel['exec_count'][:,day]>=25)
        marks[good]=opening[good]
        assert len(trades[row['date']])<=10
        for tr in trades[row['date']]:
            key=codes[tr['code']]
            assert tr['side'] in ['buy','sell'] and tr['qty']>0
            if tr['side']=='buy':assert panel['eligible'][key,day-1]
            sign=1 if tr['side']=='buy' else -1
            fee=.001 if sign==1 else .003
            assert abs(tr['qty']*tr['price']*fee-tr['fee'])<1e-5
            quantity[key]+=sign*tr['qty']
            cash-=sign*tr['qty']*tr['price']+tr['fee']
        assert (quantity>=0).all() and cash>=-1
        close=panel['close'][:,day]
        good=np.isfinite(close)&(close>0)
        marks[good]=close[good]
        nav=cash+float(quantity@marks)
        assert abs(nav-row['nav'])<1e-4 and abs(cash-row['cash'])<1e-4
    return len(case['daily'])


def evaluate(g,destination,progress=lambda *_:None):
    g=normalize(g)
    np,rankdata,inputs,linear,quadratic,ACCOUNT,KIS_CACHE=legacy()
    import run_period_corrected_pit_ridge as worker
    from replay_sizing100_stock20_cal5_u100_tiny_lambda import calibrate,retention_engine
    from cached_dynamic_sector_overlay import overlay
    from .metrics import ledger,statistics_for
    from quant.timefolio_cnn_account import turnover_windows
    manifest=model_cache(g,progress)
    scores=np.load(manifest['scores_path'],mmap_mode='r')
    index=json.loads((ACCOUNT/'index.json').read_text())
    with np.load(ACCOUNT/'panel.npz') as z:panel={n:z[n] for n in z.files}
    proxy=overlay(panel)
    ranks,slopes,calibration=calibrate(scores,panel,index)
    regular=worker.with_headroom(worker.framework.replay,.85)
    engine,stats,ast_sha=retention_engine(regular,ranks,slopes)
    forecast_path=STUDY/'liquid_market19mix_latent_head_known_forecasts_v1_20261007/review.json'
    forecast=json.loads(forecast_path.read_text())
    assert sha(forecast_path)=='7333e90224e9979876fa692cee0f6088115a2997956347e0dd9637635d5e3a60'
    schedule=np.array([g['gross_high'] if r['predicted_net20'] is None or r['predicted_net20']>0 else g['gross_low'] for r in forecast['trace']])
    from .period import window
    start, requested_end = window()
    expected=[d for d in index['dates'] if start<=d<=requested_end]
    from .period import require_complete
    require_complete(expected,'kr')
    if len({d[:6] for d in expected}) != 36 or expected[0] > start[:6]+'10':
        raise ValueError('최근 36개월 계좌 입력이 부족합니다.')
    end=expected[-1]
    if end < requested_end[:6]+'20':
        raise ValueError('최근 월의 계좌 입력이 부족합니다.')
    first_signal=index['dates'].index(expected[0])-1
    if first_signal<0 or not np.isfinite(scores[:,first_signal]).any():
        raise ValueError('평가 시작 전일의 학습 점수가 없습니다.')
    progress('backtesting',dict(start=start,end=end,months=36))
    result=engine(proxy,index,scores,expected[0],end,
                  rebalance=g['rebalance'],rank_buffer=0,rebalance_band=g['band'],top_n=g['top_n'],
                  weight=g['stock_weight'],gross=.8,max_orders=10,slip=.0005,participation=.05,
                  gross_schedule=schedule,return_trades=True)
    if [r['date'] for r in result['daily']] != expected:
        raise ValueError('평가 기간 거래일이 일치하지 않습니다.')
    evaluated=ledger(dict(daily=result['daily']))
    pooled=statistics_for(evaluated)
    folds={month:statistics_for([r for r in evaluated if r['date'].startswith(month)])
           for month in sorted({d[:6] for d in expected})}
    def gate(total,minimum):
        return pooled['sharpe'] is not None and pooled['sharpe']>total and all(
            f['sharpe'] is not None and f['sharpe']>minimum for f in folds.values())
    reg=dict(pooled=pooled,monthly_folds=folds,numerical_record_threshold=gate(2,1),numerical_stop_threshold=gate(3,1.5))
    turns=turnover_windows(result['daily'])
    case=dict(phase=0,evaluation=reg,account_metrics=result['metrics'],daily=result['daily'],
              trades=result['trades'],turnover_windows=turns,
              turnover_pass=not any(t['four_violation_screen_failed'] for t in turns))
    daily_checks=verify_book(case,proxy,index)
    cases=[case]
    report=dict(evaluation_start=start,evaluation_end=end,requested_end=requested_end,evaluation_months=36,status='local_genomic_account_backtest_and_independent_longonly_nav_passed',genome=g,
                cases=cases,model_manifest=str(RUNS/'models'/manifest['model_id']/'manifest.json'),
                model_id=manifest['model_id'],private_engine_ast_sha256=ast_sha,
                cash_forecast_sha256=sha(forecast_path),gross_schedule=schedule.tolist(),daily_rows_verified=daily_checks,
                model_proofs=manifest['proofs'],source_sha256=sha(Path(__file__)),new_stock_training=manifest['provenance'],
                no_broker_orders=True,shared_input_writes=0,input_copies=0,independent_holdout=False,contest_certified=False,
                stable_profit_verified=False,stop_for_validation=any(c['evaluation']['numerical_stop_threshold'] for c in cases),
                record_candidates=[c['phase'] for c in cases if c['evaluation']['numerical_record_threshold']],
                limitations=['Same historical development sample; 36 calendar months, through the last available session.',
                             'Proxy GICS sector caps, historical eligibility, financial vintages, corporate actions and execution; cash dividends unreconciled.'])
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/'review.json',report)
    return report

