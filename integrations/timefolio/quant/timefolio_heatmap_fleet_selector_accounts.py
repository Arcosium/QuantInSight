"""Causal monthly selectors in a continuous audited long-only account."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np

from quant import timefolio_heatmap_price_cash_accounts as engine
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_fleet_accounts import load_market, policy_grid, account_id
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_gpu_worker import digest, write, verify
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks

from quant.timefolio_heatmap_fleet_selector import MEMBERS, rules, identity, hypotheses, monthly_scores
from quant.timefolio_heatmap_fleet_blend import source_names, identity as blend_identity


def read(path): return json.loads(path.read_text())


def units(pools):
    return [dict(id=f"{pool}_{rule['id']}_{objective}_{member}", pool=pool, rule=rule,
                 objective=objective, member=member)
            for pool, components in pools.items() for rule in rules(components)
            for objective in ['trained','untrained'] for member in MEMBERS]


def warmup_prefix(result):
    return {name:[r for r in result[name] if r['date']<'20260201'] for name in ['daily','trades']}


def shadow_for(components, objective, member, policy, cache):
    selected={}
    for case in components:
        names, values=cache[case]
        key=account_id(f'flt_{case}_{objective}_{member}',policy)
        assert names.count(key)==1
        selected[case]=values[:,names.index(key)]
    return selected


def run_unit(market, root, output, unit):
    import torch
    torch.set_num_threads(1)
    spec = read(root/'execution_registration.json')
    engine.SHARED_STOP = Path(spec['shared_stop'])
    design = read(root/'design_registration.json')
    expected = read(root/'input_hashes.json')
    started = time.monotonic()
    target = output/'units'/unit['id']; target.mkdir(parents=True, exist_ok=True)
    pool, objective, member = (unit[k] for k in ['pool','objective','member'])
    components = design['pools'][pool]
    names = source_names(components, objective, member)
    matrices = {}; hashes = {}
    for name in names:
        path = root/'scores'/(name+'.npy')
        assert digest(path) == expected[str(path.relative_to(root))]
        hashes[name] = digest(path)
        matrices[name] = np.load(path, allow_pickle=False)
    raw, ix, release, _, _ = load_market(market)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    shadow_dates = read(root/'shadow_dates.json')
    assert shadow_dates == [d for d in ix['dates'] if '20260101'<=d<='20260923']
    shadow_cache={}
    for case in components:
        folder=root/'shadows'/case
        for name in ['account_ids.json','daily_returns.npy']:
            assert digest(folder/name)==expected[str((folder/name).relative_to(root))]
        ids=read(folder/'account_ids.json');values=np.load(folder/'daily_returns.npy',allow_pickle=False)
        assert values.shape==(179,128) and len(set(ids))==len(ids)==128 and np.isfinite(values).all()
        shadow_cache[case]=(ids,values)
    rows = []; rule=unit['rule']
    assert rule in rules(components)
    for policy in policy_grid():
        if not engine.can_start(output): break
        key = identity(pool, rule['id'], objective, member, policy)
        folder = target/key; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            receipt = read(folder/'complete.json')
            for name, sha in receipt['hashes'].items(): assert digest(folder/name) == sha
            rows.append(read(folder/'summary.json')); continue
        shadows=shadow_for(components,objective,member,policy,shadow_cache)
        matrix,choices=monthly_scores(matrices,raw['eligible'],ix['dates'],components,objective,member,
            shadows,shadow_dates,lookback=rule['lookback'],count=rule['count'])
        assert len(choices)==9 and all(c['last_observation'] is None or c['last_observation']<c['first_execution'] for c in choices)
        np.save(folder/'scores.npy',matrix);write(folder/'choices.json',choices)
        held={policy['refresh']:snapshot_scores(matrix,raw['eligible'],ix['dates'],'20260101','20260923',policy['refresh'])}
        panel, result, plans = engine.simulate(raw, ix, release, caps, held, policy)
        write(folder/'portfolio.json', result)
        monthly = monthly_target(result['daily']); write(folder/'monthly.json', monthly)
        verdict = engine.publish_threshold(output, key, monthly)
        checked = engine.audit(raw, panel, ix, release, caps, result, plans, key, policy, None)
        baseline=blend_identity(pool,'mean',objective,member,policy)
        source=root/'fixed_blends'/(baseline+'.json')
        assert digest(source)==expected[str(source.relative_to(root))]
        parity=engine.compare_nested(warmup_prefix(result),warmup_prefix(read(source)))
        checked['warmup_parity']=dict(id=baseline,**parity)
        write(folder/'audit.json', checked)
        blocks = calendar_blocks(result['daily'])
        row = dict(id=key, pool=pool, method=rule['id'], objective=objective, member=member, nonimage=False,
            lookback=rule['lookback'],selected_count=rule['count'],components=components,**policy,**result['metrics'],**blocks,
            positive_blocks=sum(v>0 for v in blocks.values()),**verdict,
            full_contest_compliance_certified=False,independent_confirmation=False)
        write(folder/'summary.json', row)
        write(folder/'complete.json', dict(hashes={p.name:digest(p) for p in folder.iterdir() if p.is_file()},
            independent_account_arithmetic_passed=True,warmup_parity_passed=True,final_validation_passed=False))
        rows.append(row)
        write(target/'progress.json',dict(at=time.time(),completed=len(rows),last_id=key,
            seconds=time.monotonic()-started,stopped=not engine.can_start(output)))
    write(target/'source_scores.json',dict(source_hashes=hashes,direct_seed_components=True,
        expected_components=len(names),rule=rule,no_new_fit=True))
    write(target/'summary.json', rows)
    complete = len(rows)==16
    write(target/('complete.json' if complete else 'paused.json'),dict(at=time.time(),accounts=len(rows),
        completed=complete,stopped=not engine.can_start(output),seconds=time.monotonic()-started,
        summary_sha256=digest(target/'summary.json'),source_scores_sha256=digest(target/'source_scores.json')))
    return dict(id=unit['id'],accounts=len(rows),completed=complete,seconds=time.monotonic()-started)


def queue(market, root, output, workers):
    assert 1<=workers<=16
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock=(output/'queue.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(root/'execution_registration.json');design=read(root/'design_registration.json')
    engine.SHARED_STOP=Path(spec['shared_stop'])
    assert str(engine.SHARED_STOP)=='/workspace/arctrade_cash_control_v1/output/STOP.json'
    assert engine.SHARED_STOP.parent.stat().st_dev==output.stat().st_dev
    assert read(engine.SHARED_STOP.parent/'predecessors_complete.json')['stop_accounts']==0
    assert digest(market/'manifest.json')==spec['market_manifest_sha256'];verify(market)
    for name,sha in read(root/'input_hashes.json').items():assert digest(root/name)==sha
    assert read(root/'hypotheses.json')==[list(r) for r in hypotheses(design['pools'])]
    pending=units(design['pools']);completed=[]
    for unit in list(pending):
        done=output/'units'/unit['id']/'complete.json'
        if done.exists():
            receipt=read(done);assert receipt['completed'] and receipt['accounts']==16
            assert digest(done.parent/'summary.json')==receipt['summary_sha256']
            completed.append(dict(id=unit['id'],**receipt));pending.remove(unit)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        active={}
        while pending or active:
            while pending and len(active)<workers and engine.can_start(output):
                unit=pending.pop(0);active[executor.submit(run_unit,market,root,output,unit)]=unit
            if not active:break
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                unit=active.pop(future)
                try:completed.append(future.result())
                except Exception as exc:
                    failure=dict(at=time.time(),unit=unit['id'],error=repr(exc));write(output/'failure.json',failure)
                    if engine.can_start(output):write(output/'STOP.json',dict(**failure,reason='audit_or_worker_failure'))
                    engine.propagate_stop(output);raise
            write(output/'status.json',dict(at=time.time(),state='paused' if not engine.can_start(output) else 'running',
                completed=completed,running=list(active.values()),pending=len(pending),workers=workers,
                registration_sha256=digest(root/'execution_registration.json')))
    write(output/('complete.json' if engine.can_start(output) and not pending else 'paused.json'),
        dict(at=time.time(),completed=completed,remaining=len(pending),stopped=not engine.can_start(output),
             inference_pending=True,final_validation_passed=False,reserved_outcomes_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['market','root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();queue(a.market,a.root,a.output,a.workers)
