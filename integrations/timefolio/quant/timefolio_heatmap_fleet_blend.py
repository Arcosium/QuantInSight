"""Frozen cross-heatmap rank consensus, using the audited long-only ledger."""
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

MEMBERS = ['seed17', 'seed29', 'seed43', 'ensemble']
METHODS = ['mean', 'median']


def read(path): return json.loads(path.read_text())


def identity(pool, method, objective, member, policy):
    return account_id(f'blend_{pool}_{method}_{objective}_{member}', policy)


def units(pools):
    return [dict(id=f'{pool}_{objective}_{member}', pool=pool, objective=objective, member=member)
            for pool in pools for objective in ['trained', 'untrained'] for member in MEMBERS]


def hypotheses(pools):
    family = []
    for pool, components in pools.items():
        for method in METHODS:
            for member in MEMBERS:
                for policy in policy_grid():
                    trained = identity(pool, method, 'trained', member, policy)
                    untrained = identity(pool, method, 'untrained', member, policy)
                    family.extend([(trained, 'cash', None), (trained, 'matched_untrained_blend', untrained),
                        (trained, 'online_nonimage', account_id('online_nonimage', policy)),
                        (untrained, 'cash', None)])
                    family.extend((trained, 'component_'+case, account_id(f'flt_{case}_trained_{member}', policy))
                                  for case in components)
    assert len(family) == len({(r[0], r[1]) for r in family})
    return family


def source_names(components, objective, member):
    if objective not in ['trained', 'untrained'] or member not in MEMBERS:
        raise ValueError('Registered objective and seed member required')
    if len(components) != len(set(components)) or not components:
        raise ValueError('Unique nonempty component cases required')
    seeds = MEMBERS[:3] if member == 'ensemble' else [member]
    return [f'flt_{case}_{objective}_{seed}' for case in components for seed in seeds]


def combine(matrices, eligible, method):
    if method not in METHODS: raise ValueError('Unregistered consensus method')
    return rank_consensus(matrices, {key:'component' for key in matrices}, eligible, method)


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
    rows = []
    for method in METHODS:
        if not engine.can_start(output): break
        matrix = combine(matrices, raw['eligible'], method)
        np.save(target/('scores_'+method+'.npy'), matrix)
        held = {r:snapshot_scores(matrix, raw['eligible'], ix['dates'], '20260101','20260923', r) for r in [1,5]}
        for policy in policy_grid():
            if not engine.can_start(output): break
            key = identity(pool, method, objective, member, policy)
            folder = target/key; folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():
                receipt = read(folder/'complete.json')
                for name, sha in receipt['hashes'].items(): assert digest(folder/name) == sha
                rows.append(read(folder/'summary.json')); continue
            panel, result, plans = engine.simulate(raw, ix, release, caps, held, policy)
            write(folder/'portfolio.json', result)
            monthly = monthly_target(result['daily']); write(folder/'monthly.json', monthly)
            verdict = engine.publish_threshold(output, key, monthly)
            checked = engine.audit(raw, panel, ix, release, caps, result, plans, key, policy, None)
            write(folder/'audit.json', checked)
            blocks = calendar_blocks(result['daily'])
            row = dict(id=key, pool=pool, method=method, objective=objective, member=member, nonimage=False,
                components=components, **policy, **result['metrics'], **blocks,
                positive_blocks=sum(v>0 for v in blocks.values()), **verdict,
                full_contest_compliance_certified=False, independent_confirmation=False)
            write(folder/'summary.json', row)
            write(folder/'complete.json', dict(hashes={p.name:digest(p) for p in folder.iterdir() if p.is_file()},
                independent_account_arithmetic_passed=True, final_validation_passed=False))
            rows.append(row)
            write(target/'progress.json', dict(at=time.time(), completed=len(rows), last_id=key,
                seconds=time.monotonic()-started, stopped=not engine.can_start(output)))
    write(target/'source_scores.json', dict(source_hashes=hashes, direct_seed_components=True,
          expected_components=len(names), methods=METHODS, no_new_fit=True))
    write(target/'summary.json', rows)
    complete = len(rows)==32
    write(target/('complete.json' if complete else 'paused.json'), dict(at=time.time(), accounts=len(rows),
        completed=complete, stopped=not engine.can_start(output), seconds=time.monotonic()-started,
        summary_sha256=digest(target/'summary.json'), source_scores_sha256=digest(target/'source_scores.json'),
        combined_scores={p.name:digest(p) for p in target.glob('scores_*.npy')}))
    return dict(id=unit['id'], accounts=len(rows), completed=complete, seconds=time.monotonic()-started)


def queue(market, root, output, workers):
    assert 1<=workers<=8
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
            receipt=read(done);assert receipt['completed'] and receipt['accounts']==32
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
