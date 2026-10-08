"""Run the registered opening-score variants using the unchanged long-only ledger."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time
import numpy as np
from quant import timefolio_heatmap_price_cash_accounts as engine
from quant.timefolio_heatmap_fleet_accounts import load_market, policy_grid, account_id
from quant.timefolio_heatmap_fleet_breadth_accounts import units
from quant.timefolio_heatmap_opening_rank import VARIANTS, transform, identity, hypotheses
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_gpu_worker import digest, write, verify
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks


def read(path):return json.loads(path.read_text())


def run_unit(market, root, source, output, unit):
    import torch
    torch.set_num_threads(1)
    spec=read(root/'registration.json');engine.SHARED_STOP=Path(spec['shared_stop'])
    target=output/'units'/unit['id'];target.mkdir(parents=True,exist_ok=True)
    raw,ix,release,_,_=load_market(market);caps=historical_stock_caps(ix['codes'],ix['dates'])
    model='online_nonimage' if unit['nonimage'] else 'flt_'+unit['id']
    path=market/'online_nonimage.npy' if unit['nonimage'] else source/'sources'/unit['case']/'scores'/(model+'.npy')
    matrix=np.load(path,allow_pickle=False)
    original={r:snapshot_scores(matrix,raw['eligible'],ix['dates'],'20260101','20260923',r) for r in [1,5]}
    write(target/'source_score.json',dict(source_score_sha256=digest(path),model=model,no_new_neural_fit=True))
    zero={r:transform(original[r],raw,ix['dates'],'zero') for r in [1,5]}
    zero_parities=[]
    for r,(score,proof) in zero.items():
        np.save(target/f'zero_refresh{r}_scores.npy',score)
        write(target/f'zero_refresh{r}_causality.json',proof)
    parities=[];online={r['summary']['id']:r for r in read(source/'online.json')} if unit['nonimage'] else None
    for policy in policy_grid():
        if not engine.can_start(output):break
        _,result,_=engine.simulate(raw,ix,release,caps,original,policy)
        key=account_id(model,policy)
        if unit['nonimage']:
            nav=np.asarray([1e9]+[d['nav'] for d in result['daily']])
            error=float(np.max(np.abs(nav[1:]/nav[:-1]-1-online[key]['daily_returns'])))
            assert error<1e-12
            proof=dict(maximum_absolute_float_error=0.,saved_return_maximum_error=error)
        else:
            proof=engine.compare_nested(result,read(source/'sources'/unit['case']/'portfolios'/(key+'.json')))
        parities.append(dict(id=key,**proof))
        _,zero_result,_=engine.simulate(raw,ix,release,caps,original,policy,decision_scores=zero[policy['refresh']][0])
        zero_parities.append(dict(id=key,**engine.compare_nested(zero_result,result)))
    write(target/'baseline_parity.json',parities)
    write(target/'zero_overlay_parity.json',zero_parities)
    rows=[];derived={'zero':{str(r):dict(score_sha256=digest(target/f'zero_refresh{r}_scores.npy'),
        causality_sha256=digest(target/f'zero_refresh{r}_causality.json')) for r in [1,5]}}
    started=time.monotonic()
    for variant in VARIANTS:
        if not engine.can_start(output):break
        assert len(parities)==16
        held=original;decisions={};derived[variant]={}
        for refresh in [1,5]:
            score,proof=transform(held[refresh],raw,ix['dates'],variant)
            stem=f'{variant}_refresh{refresh}'
            np.save(target/(stem+'_scores.npy'),score)
            write(target/(stem+'_causality.json'),proof)
            derived[variant][str(refresh)]=dict(score_sha256=digest(target/(stem+'_scores.npy')),
                causality_sha256=digest(target/(stem+'_causality.json')))
            decisions[refresh]=score
        for policy in policy_grid():
            if not engine.can_start(output):break
            key=identity(model,variant,policy);folder=target/key;folder.mkdir(exist_ok=True)
            assert not (folder/'complete.json').exists(),'Never overwrite an existing opening account'
            panel,result,plans=engine.simulate(raw,ix,release,caps,held,policy,decision_scores=decisions[policy['refresh']])
            for trade in result['trades']:
                trade.update(decision_score_date=trade['date'],decision_last_input_time='09:00:00',execution_not_before='09:05:00')
            write(folder/'portfolio.json',result)
            monthly=monthly_target(result['daily']);write(folder/'monthly.json',monthly)
            verdict=engine.publish_threshold(output,key,monthly)
            checked=engine.audit(raw,panel,ix,release,caps,result,plans,key,policy,None)
            checked['opening_decision_causality_passed']=True
            for trade in result['trades']:
                d=ix['dates'].index(trade['date']);origin=int(held[policy['refresh']][1][d-1])
                assert trade['portfolio_score_date']==ix['dates'][origin] and origin<=d-1
                assert trade['decision_score_date']==ix['dates'][d] and trade['decision_last_input_time']<trade['execution_not_before']
            write(folder/'audit.json',checked)
            blocks=calendar_blocks(result['daily'])
            row=dict(id=key,case=unit['case'],objective=unit['objective'],member=unit['member'],
                nonimage=unit['nonimage'],variant=variant,**policy,**result['metrics'],**blocks,
                positive_blocks=sum(v>0 for v in blocks.values()),**verdict,
                independent_confirmation=False,full_contest_compliance_certified=False)
            write(folder/'summary.json',row)
            write(folder/'complete.json',dict(hashes={p.name:digest(p) for p in folder.iterdir() if p.is_file()},
                original_account_parity_passed=True,zero_overlay_parity_passed=True,opening_decision_causality_passed=True,independent_account_arithmetic_passed=True,final_validation_passed=False))
            rows.append(row)
            write(target/'progress.json',dict(at=time.time(),accounts=len(rows),last_id=key))
    write(target/'summary.json',rows);write(target/'derived_hashes.json',derived)
    completed=len(rows)==64
    receipt=dict(at=time.time(),accounts=len(rows),completed=completed,stopped=not engine.can_start(output),
        seconds=time.monotonic()-started,original_accounts_reproduced=len(parities),zero_overlay_accounts_reproduced=len(zero_parities),
        summary_sha256=digest(target/'summary.json'),source_score_sha256=digest(target/'source_score.json'),
        baseline_parity_sha256=digest(target/'baseline_parity.json'),zero_overlay_parity_sha256=digest(target/'zero_overlay_parity.json'),derived_hashes_sha256=digest(target/'derived_hashes.json'))
    write(target/('complete.json' if completed else 'paused.json'),receipt)
    return dict(id=unit['id'],**receipt)


def run(market,root,source,output,workers):
    import fcntl
    assert 1<=workers<=8
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(root/'registration.json');engine.SHARED_STOP=Path(spec['shared_stop'])
    assert not engine.SHARED_STOP.exists()
    verify(market)
    assert digest(source/'input_hashes.json')==spec['source_manifest_sha256']
    for name,h in read(source/'input_hashes.json').items():assert digest(source/name)==h
    for name,h in read(root/'runtime_hashes.json').items():assert digest(root/'runtime'/name)==h
    assert read(root/'hypotheses.json')==[list(r) for r in hypotheses(spec['source_cases'])]
    pending=units(spec['source_cases']);completed=[]
    assert len(pending)==73 and not list((output/'units').glob('*/complete.json'))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        active={}
        while pending or active:
            while pending and len(active)<workers and engine.can_start(output):
                unit=pending.pop(0);active[ex.submit(run_unit,market,root,source,output,unit)]=unit
            if not active:break
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                unit=active.pop(future)
                try:completed.append(future.result())
                except Exception as exc:
                    failure=dict(at=time.time(),unit=unit,error=repr(exc))
                    write(output/'failure.json',failure)
                    if engine.can_start(output):write(output/'STOP.json',dict(failure,reason='audit_or_worker_failure'))
                    engine.propagate_stop(output);raise
            write(output/'status.json',dict(at=time.time(),completed=completed,running=list(active.values()),
                pending=len(pending),workers=workers,stopped=not engine.can_start(output)))
    write(output/('complete.json' if not pending and engine.can_start(output) else 'paused.json'),
        dict(at=time.time(),completed=completed,remaining=len(pending),stopped=not engine.can_start(output)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['market','root','source','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    run(a.market,a.root,a.source,a.output,a.workers)
