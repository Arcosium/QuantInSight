"""Replay registered breadth policies after full original-account parity."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np

from quant import timefolio_heatmap_price_cash_accounts as engine
from quant.timefolio_heatmap_fleet_accounts import load_market, policy_grid, account_id
from quant.timefolio_heatmap_fleet_breadth import MEMBERS, policies, identity, hypotheses
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_gpu_worker import digest, write, verify
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks


def read(path):
    return json.loads(path.read_text())


def units(cases):
    rows = [dict(id=f'{case}_{objective}_{member}', case=case, objective=objective,
                 member=member, nonimage=False)
            for case in cases for objective in ['trained', 'untrained'] for member in MEMBERS]
    return rows + [dict(id='online_nonimage', case=None, objective='nonimage', member='online', nonimage=True)]


def simulate(raw, ix, release, caps, held, policy):
    """Only top_n and weight are new parameters; omitted values reproduce v1."""
    panel = dict(raw)
    if policy['ceiling'] == 'research20':
        panel['sector_cap'] = np.minimum(raw['sector_cap'], .2)
    alpha, origins = held[policy['refresh']]
    result = engine.replay(panel, ix, alpha, '20260101', '20260923',
        top_n=policy.get('top_n', 12), weight=policy.get('weight', .05),
        max_orders=policy['max_orders'], rank_buffer=policy['buffer'], rebalance=5,
        rebalance_band=.0005, return_trades=True, return_plans=True, planning_price='open',
        action_release_dates=release, stock_cap_schedule=caps, locked_repair=True)
    dates = {d: i for i, d in enumerate(ix['dates'])}
    for trade in result['trades']:
        day = dates[trade['signal_date']]
        assert 0 <= origins[day] <= day
        trade['portfolio_score_date'] = ix['dates'][origins[day]]
    result['metrics'].update(assess_weeks(result))
    plans, result['plans'] = result['plans'], []
    return panel, result, plans


def run_unit(market, root, output, unit):
    import torch
    torch.set_num_threads(1)
    spec = read(root/'execution_registration.json')
    engine.SHARED_STOP = Path(spec['shared_stop'])
    expected = read(root/'input_hashes.json')
    started = time.monotonic()
    target = output/'units'/unit['id']; target.mkdir(parents=True, exist_ok=True)
    raw, ix, release, _, _ = load_market(market)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    model = 'online_nonimage' if unit['nonimage'] else 'flt_' + unit['id']
    source = market/'online_nonimage.npy' if unit['nonimage'] else root/'sources'/unit['case']/'scores'/(model+'.npy')
    if not unit['nonimage']:
        assert digest(source) == expected[str(source.relative_to(root))]
    matrix = np.load(source, allow_pickle=False)
    held = {r: snapshot_scores(matrix, raw['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
    source_proof = dict(source_score_sha256=digest(source), model=model, source=str(source), no_new_fit=True)
    write(target/'source_score.json', source_proof)
    online = {r['summary']['id']: r for r in read(root/'online.json')} if unit['nonimage'] else None
    parities = []
    for policy in policy_grid():
        if not engine.can_start(output): break
        key = account_id(model, policy)
        _, original, _ = simulate(raw, ix, release, caps, held, policy)
        if unit['nonimage']:
            _, baseline, _ = engine.simulate(raw, ix, release, caps, held, policy)
            compare = engine.compare_nested(original, baseline)
            nav = np.asarray([1e9] + [d['nav'] for d in original['daily']])
            error = float(np.max(np.abs(nav[1:] / nav[:-1] - 1 - np.asarray(online[key]['daily_returns']))))
            assert error < 1e-12
            compare['saved_return_maximum_error'] = error
        else:
            path = root/'sources'/unit['case']/'portfolios'/(key+'.json')
            assert digest(path) == expected[str(path.relative_to(root))]
            compare = engine.compare_nested(original, read(path))
        parities.append(dict(id=key, **compare))
    write(target/'baseline_parity.json', parities)
    rows = []
    for policy in policies():
        if not engine.can_start(output): break
        assert len(parities) == 16
        key = identity(unit['id'], policy)
        folder = target/key; folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            for name, sha in read(folder/'complete.json')['hashes'].items():
                assert digest(folder/name) == sha
            rows.append(read(folder/'summary.json')); continue
        panel, result, plans = simulate(raw, ix, release, caps, held, policy)
        write(folder/'portfolio.json', result)
        monthly = monthly_target(result['daily']); write(folder/'monthly.json', monthly)
        verdict = engine.publish_threshold(output, key, monthly)
        checked = engine.audit(raw, panel, ix, release, caps, result, plans, key, policy, None)
        write(folder/'audit.json', checked)
        blocks = calendar_blocks(result['daily'])
        row = dict(id=key, case=unit['case'], objective=unit['objective'], member=unit['member'],
            nonimage=unit['nonimage'], **policy, **result['metrics'], **blocks,
            positive_blocks=sum(v>0 for v in blocks.values()), **verdict,
            full_contest_compliance_certified=False, independent_confirmation=False)
        write(folder/'summary.json', row)
        write(folder/'complete.json', dict(hashes={p.name:digest(p) for p in folder.iterdir() if p.is_file()},
            independent_account_arithmetic_passed=True, original_account_parity_passed=True, final_validation_passed=False))
        rows.append(row)
        write(target/'progress.json', dict(at=time.time(), completed=len(rows), last_id=key,
            seconds=time.monotonic()-started, stopped=not engine.can_start(output)))
    write(target/'summary.json', rows)
    complete = len(rows) == 48
    write(target/('complete.json' if complete else 'paused.json'), dict(at=time.time(), accounts=len(rows),
        completed=complete, stopped=not engine.can_start(output), seconds=time.monotonic()-started,
        summary_sha256=digest(target/'summary.json'), source_score_sha256=digest(target/'source_score.json'),
        baseline_parity_sha256=digest(target/'baseline_parity.json'), original_accounts_reproduced=len(parities)))
    return dict(id=unit['id'], accounts=len(rows), completed=complete, seconds=time.monotonic()-started)


def queue(market, root, output, workers):
    assert 1 <= workers <= 16
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/'queue.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = read(root/'execution_registration.json'); design = read(root/'design_registration.json')
    engine.SHARED_STOP = Path(spec['shared_stop'])
    assert str(engine.SHARED_STOP) == '/workspace/arctrade_cash_control_v1/output/STOP.json'
    assert engine.SHARED_STOP.parent.stat().st_dev == output.stat().st_dev
    assert read(engine.SHARED_STOP.parent/'predecessors_complete.json')['stop_accounts'] == 0
    assert digest(market/'manifest.json') == spec['market_manifest_sha256']; verify(market)
    for name, sha in read(root/'input_hashes.json').items(): assert digest(root/name) == sha
    assert read(root/'hypotheses.json') == [list(r) for r in hypotheses(design['source_cases'])]
    pending = units(design['source_cases']); completed = []
    for unit in list(pending):
        done = output/'units'/unit['id']/'complete.json'
        if done.exists():
            receipt = read(done); assert receipt['completed'] and receipt['accounts'] == 48
            assert digest(done.parent/'summary.json') == receipt['summary_sha256']
            completed.append(dict(id=unit['id'], **receipt)); pending.remove(unit)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        active = {}
        while pending or active:
            while pending and len(active) < workers and engine.can_start(output):
                unit = pending.pop(0); active[executor.submit(run_unit, market, root, output, unit)] = unit
            if not active: break
            done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                unit = active.pop(future)
                try: completed.append(future.result())
                except Exception as exc:
                    failure = dict(at=time.time(), unit=unit['id'], error=repr(exc)); write(output/'failure.json', failure)
                    if engine.can_start(output): write(output/'STOP.json', dict(**failure, reason='audit_or_worker_failure'))
                    engine.propagate_stop(output); raise
            write(output/'status.json', dict(at=time.time(), state='paused' if not engine.can_start(output) else 'running',
                completed=completed, running=list(active.values()), pending=len(pending), workers=workers,
                registration_sha256=digest(root/'execution_registration.json')))
    write(output/('complete.json' if engine.can_start(output) and not pending else 'paused.json'),
        dict(at=time.time(), completed=completed, remaining=len(pending), stopped=not engine.can_start(output),
             inference_pending=True, final_validation_passed=False, reserved_outcomes_read=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['market', 'root', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(); queue(a.market, a.root, a.output, a.workers)
