"""Price-only cash controls for fixed pairwise heatmaps, with cross-study stops.

All price data, model scores and unchanged ledgers come from the frozen fleet.
The scheduler checks a shared stop marker before starting every new account.
Finished ledgers are retained even when a later account triggers validation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from quant.timefolio_heatmap_fleet_accounts import (
    load_market, policy_grid, account_id, compare_nested,
)
from quant.timefolio_heatmap_cash_controls import price_schedules
from quant.timefolio_heatmap_locked_weighted_replay import replay
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_fleet_thresholds import classify
from quant.timefolio_heatmap_fleet_audit_helpers import (
    inspect_account, screen_account, plan_checks, classify_plans, summarize,
)
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_action_amendment import release_audit
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks
from quant.timefolio_heatmap_gpu_worker import digest, write

PARENT_SHA256 = 'c41457f96a53afe56e7089b09622a6ec298e42c0b24e25ee632dc5e49119672d'
SHARED_STOP = None
CONTROLS = ['trend20_60', 'breadth20', 'volatility20']
MEMBERS = ['seed17', 'seed29', 'seed43', 'ensemble']


def read(path):
    return json.loads(path.read_text())


def identity(case, objective, member, policy, control, nonimage=False):
    prefix = 'price_cash_nonimage_for_' if nonimage else 'price_cash_'
    return prefix + account_id(f'flt_{case}_{objective}_{member}', policy) + '__' + control


def units(cases):
    return [dict(id=f'{case}_{objective}_{member}', case=case, objective=objective, member=member)
            for case in cases for objective in ['trained', 'untrained'] for member in MEMBERS]


def hypotheses(cases):
    rows = []
    for unit in units(cases):
        c, o, m = (unit[k] for k in ['case', 'objective', 'member'])
        for policy in policy_grid():
            baseline = account_id(f'flt_{c}_{o}_{m}', policy)
            for control in CONTROLS:
                key = identity(c, o, m, policy, control)
                online = identity(c, o, m, policy, control, True)
                rows.extend([(key, 'cash', None), (key, 'unchanged_baseline', baseline),
                             (key, 'same_exposure_nonimage', online), (online, 'cash', None)])
                if o == 'trained':
                    rows.append((key, 'matching_untrained_pipeline', identity(c, 'untrained', m, policy, control)))
    assert len(rows) == len({(r[0], r[1]) for r in rows})
    return rows


def publish_threshold(output, key, monthly):
    """Publish a complete candidate before the cursor; never overwrite a stop."""
    verdict = classify(monthly)
    if verdict['retain_candidate']:
        record = dict(at=time.time(), id=key, monthly=monthly, **verdict,
                      independent_audit_pending=True)
        target = output / 'candidates' / (key + '.json')
        target.parent.mkdir(exist_ok=True)
        write(target, record)
        if verdict['stop_search_and_validate']:
            # A hard link publishes the fully written candidate atomically. A
            # racing worker can only create the first marker, never erase it.
            try:
                os.link(target, output / 'STOP.json')
            except FileExistsError:
                pass
            propagate_stop(output)
    return verdict


def propagate_stop(output):
    if SHARED_STOP is not None and not SHARED_STOP.exists():
        try:
            os.link(output / 'STOP.json', SHARED_STOP)
        except FileExistsError:
            pass


def can_start(output):
    return not (output / 'STOP.json').exists() and (SHARED_STOP is None or not SHARED_STOP.exists())


def closing_audit(raw, ix, result, key, policy, caps, panel, plans):
    exposure, events = inspect_account(raw, ix, result, key, policy['ceiling'], caps)
    relevant = [e for e in events if e['category'] in ['official_sector', 'small_cap']]
    screens = screen_account(relevant, result, raw, ix)
    unresolved = [e for e in screens if e['status'] == 'still_above_no_relevant_sale']
    details, available = [], []
    classified = None
    for event in unresolved:
        detail = plan_checks(event, plans, panel, ix, raw['sector_cap'])
        detail.update(next_filled_orders=sum(t['date'] == event['next_date'] for t in result['trades']),
                      max_orders=policy['max_orders'])
        details.append(detail)
        for plan in detail['sell_plans']:
            if plan['execution_price_missing'] and plan['available_capacity'] == 0:
                continue
            if classified is None:
                classified = {(v['date'], v['code'], v['side']): v for v in
                              classify_plans(panel, ix, plans, result['trades'], policy['max_orders'])}
            row = classified[event['next_date'], plan['code'], 'sell']
            assert row['filled_qty'] == 0
            available.append(dict(id=key, previous_date=event['date'], category=event['category'], **row))
    unplanned = [e for e in details if not e['sell_plans'] and
                 (e['category'] == 'small_cap' or not e['prior_weight_within_next_limit'])]
    return dict(exposure=exposure, events=events, screen=screens, unresolved_details=details,
                summary=summarize(screens, details), available_repair_details=available,
                available_repair_stage_counts=dict(Counter(r['stage'] for r in available)),
                unplanned_exceptions=unplanned, full_contest_compliance_certified=False)


def simulate(raw, ix, release, caps, held, policy, schedule=None, *, decision_scores=None):
    panel = dict(raw)
    if policy['ceiling'] == 'research20':
        panel['sector_cap'] = np.minimum(raw['sector_cap'], .2)
    alpha, origins = held[policy['refresh']]
    kw = {} if schedule is None else dict(gross_schedule=schedule, weight_schedule=.05 * schedule / .6)
    if decision_scores is not None:kw["decision_scores"] = decision_scores
    result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12, weight=.05,
        max_orders=policy['max_orders'], rank_buffer=policy['buffer'], rebalance=5,
        rebalance_band=.0005, return_trades=True, return_plans=True, planning_price='open',
        action_release_dates=release, stock_cap_schedule=caps, locked_repair=True, **kw)
    dates = {d: i for i, d in enumerate(ix['dates'])}
    for trade in result['trades']:
        day = dates[trade['signal_date']]
        assert 0 <= origins[day] <= day
        trade['portfolio_score_date'] = ix['dates'][origins[day]]
    result['metrics'].update(assess_weeks(result))
    plans, result['plans'] = result['plans'], []
    return panel, result, plans


def audit(raw, panel, ix, release, caps, result, plans, key, policy, schedule):
    report = audit_fills(panel, ix, result)
    report.update(additional_checks(panel, ix, result, schedule, max_orders=policy['max_orders']))
    report['announced_action_errors'] = release_audit(panel, ix, result, release)
    report['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
    assert not any(report[k] for k in ['post_buy_limit_violations', 'additional_errors', 'announced_action_errors'])
    assert not report['dated_hynix_audit']['post_buy_limit_errors']
    assert report['maximum_nav_reconstruction_error_krw'] < .01
    report['closing'] = closing_audit(raw, ix, result, key, policy, caps, panel, plans)
    return report


def run_unit(market, root, output, unit):
    import torch
    global SHARED_STOP
    SHARED_STOP = Path(read(root / 'execution_registration.json')['shared_stop'])
    torch.set_num_threads(1)
    started = time.monotonic()
    target = output / 'units' / unit['id']
    target.mkdir(parents=True, exist_ok=True)
    c, o, m = (unit[k] for k in ['case', 'objective', 'member'])
    source = root / 'sources' / c
    model = f'flt_{c}_{o}_{m}'
    hashes = read(root / 'input_hashes.json')
    def frozen(path):
        assert digest(path) == hashes[str(path.relative_to(root))]
        return path
    matrix = np.load(frozen(source / 'scores' / (model + '.npy')), allow_pickle=False)
    raw, ix, release, _, _ = load_market(market)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    online = np.load(market / 'online_nonimage.npy', allow_pickle=False)
    def snapshots(a):
        return {r: snapshot_scores(a, raw['eligible'], ix['dates'], '20260101', '20260923', r) for r in [1, 5]}
    held, online_held = snapshots(matrix), snapshots(online)
    controls = price_schedules(raw)
    np.savez(target / 'schedules.npz', **controls)
    rows, parity = [], []
    for policy in policy_grid():
        if not can_start(output): break
        base_id = account_id(model, policy)
        _, baseline, _ = simulate(raw, ix, release, caps, held, policy)
        expected = read(frozen(source / 'portfolios' / (base_id + '.json')))
        parity.append(dict(id=base_id, **compare_nested(baseline, expected)))
        for control in CONTROLS:
            for nonimage in [False, True]:
                if not can_start(output): break
                key = identity(c, o, m, policy, control, nonimage)
                folder = target / key
                folder.mkdir(exist_ok=True)
                done = folder / 'complete.json'
                if done.exists():
                    complete = read(done)
                    for name, sha in complete['hashes'].items(): assert digest(folder / name) == sha
                    rows.append(read(folder / 'summary.json'))
                    continue
                panel, result, plans = simulate(raw, ix, release, caps, online_held if nonimage else held,
                                                policy, controls[control])
                write(folder / 'portfolio.json', result)
                monthly = monthly_target(result['daily'])
                write(folder / 'monthly.json', monthly)
                verdict = publish_threshold(output, key, monthly)
                checked = audit(raw, panel, ix, release, caps, result, plans, key, policy, controls[control])
                write(folder / 'audit.json', checked)
                blocks = calendar_blocks(result['daily'])
                row = dict(id=key, case=c, objective=o, member=m, nonimage=nonimage, control=control,
                    **policy, **result['metrics'], **blocks, positive_blocks=sum(v > 0 for v in blocks.values()),
                    **verdict, full_contest_compliance_certified=False, independent_confirmation=False)
                write(folder / 'summary.json', row)
                write(done, dict(hashes={p.name: digest(p) for p in folder.iterdir() if p.is_file()},
                                  independent_account_arithmetic_passed=True, final_validation_passed=False))
                rows.append(row)
                write(target / 'progress.json', dict(at=time.time(), completed=len(rows), last_id=key,
                      seconds=time.monotonic()-started, stopped=not can_start(output)))
    write(target / 'baseline_parity.json', parity)
    write(target / 'summary.json', rows)
    finished = len(rows) == 96
    write(target / ('complete.json' if finished else 'paused.json'), dict(at=time.time(), accounts=len(rows),
        completed=finished, stopped=not can_start(output), seconds=time.monotonic()-started,
        summary_sha256=digest(target / 'summary.json'), baseline_parities=len(parity)))
    return dict(id=unit['id'], accounts=len(rows), completed=finished, seconds=time.monotonic()-started)


def queue(market, root, output, workers, unit_ids=None):
    assert 1 <= workers <= 8
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registration = read(root / 'execution_registration.json')
    global SHARED_STOP
    SHARED_STOP = Path(registration['shared_stop'])
    assert str(SHARED_STOP) == '/workspace/arctrade_cash_control_v1/output/STOP.json'
    assert SHARED_STOP.parent.stat().st_dev == output.stat().st_dev
    assert digest(market / 'manifest.json') == registration['market_manifest_sha256']
    for name, sha in read(market / 'manifest.json').items(): assert digest(market / name) == sha
    for name, sha in read(root / 'input_hashes.json').items(): assert digest(root / name) == sha
    assert read(root / 'hypotheses.json') == [list(r) for r in hypotheses(registration['cases'])]
    pending = units(registration['cases'])
    if unit_ids is not None:
        assert len(unit_ids) == len(set(unit_ids)) and set(unit_ids) <= {u['id'] for u in pending}
        pending = [u for u in pending if u['id'] in unit_ids]
    completed = []
    for u in list(pending):
        done = output / 'units' / u['id'] / 'complete.json'
        if done.exists():
            receipt = read(done)
            assert receipt['accounts'] == 96 and receipt['completed']
            assert digest(done.parent / 'summary.json') == receipt['summary_sha256']
            completed.append(dict(id=u['id'], **receipt)); pending.remove(u)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        active = {}
        while pending or active:
            while pending and len(active) < workers and can_start(output):
                unit = pending.pop(0)
                active[executor.submit(run_unit, market, root, output, unit)] = unit
            if not active: break
            done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                unit = active.pop(future)
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failure = dict(at=time.time(), unit=unit['id'], error=repr(exc))
                    write(output / 'failure.json', failure)
                    if can_start(output): write(output / 'STOP.json', dict(**failure, reason='audit_or_worker_failure'))
                    propagate_stop(output)
                    raise
            write(output / 'status.json', dict(at=time.time(), state='paused' if not can_start(output) else 'running',
                completed=completed, running=list(active.values()), pending=len(pending), workers=workers,
                registration_sha256=digest(root / 'execution_registration.json')))
    write(output / ('complete.json' if can_start(output) and not pending else 'paused.json'),
        dict(at=time.time(), completed=completed, remaining=len(pending), stopped=not can_start(output),
             inference_pending=True, final_validation_passed=False, reserved_outcomes_read=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['market', 'root', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--unit', action='append')
    a = p.parse_args()
    queue(a.market, a.root, a.output, a.workers, a.unit)
