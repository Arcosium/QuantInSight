"""Independent account arithmetic and unchanged closing followup on each Pod."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from quant.timefolio_heatmap_fleet_accounts import (
    load_market, policy_grid, account_id, compare_nested,
)
from quant.timefolio_heatmap_fleet_lab import cases, MEMBERS
from quant.timefolio_heatmap_gpu_worker import digest, write, verify
from quant.timefolio_heatmap_fleet_audit_helpers import (
    inspect_account, screen_account, plan_checks, classify_plans, summarize,
)
from quant.timefolio_heatmap_locked_replay import replay
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_paper_benchmark import nav_metrics, distance
from quant.timefolio_heatmap_fleet_metrics import monthly_target


def validate_case(market_root, source, output, reference):
    """Read a completed case; write a separate audit, never change its ledger."""
    assert not output.exists()
    complete = json.loads((source / 'complete.json').read_text())
    case = complete['case']
    assert case in {c['id'] for c in cases()} and complete['accounts'] == 128
    assert complete['package_sha256'] == digest(market_root / 'manifest.json')
    manifest = json.loads((source / 'artifact_hashes.json').read_text())
    for name, sha in manifest.items():
        p = source / name
        assert p.resolve().is_relative_to(source.resolve()) and digest(p) == sha
    rows = json.loads((source / 'summary.json').read_text())
    models = [f'flt_{case}_{objective}_{member}'
              for objective in ['trained', 'untrained'] for member in MEMBERS]
    expected = {account_id(m, p) for m in models for p in policy_grid()}
    assert len(rows) == len(expected) == 128 and {r['id'] for r in rows} == expected
    raw, ix, release, _, _ = load_market(market_root)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    expected_dates = [d for d in ix['dates'] if d >= '20260101']
    assert len(expected_dates) == 179
    daily_returns = []
    exposure, events, screens, details, reproduced, available, proximity = [], [], [], [], [], [], []
    held = {}
    max_metric_error = 0.
    for row in rows:
        key, model = row['id'], row['model']
        result = json.loads((source / 'portfolios' / (key + '.json')).read_text())
        assert [d['date'] for d in result['daily']] == expected_dates
        nav = np.array([1e9] + [d['nav'] for d in result['daily']])
        metrics = nav_metrics(nav[1:])
        for field, old in [('net_return', 'return'), ('mdd', 'mdd'), ('sharpe', 'sharpe')]:
            error = abs(metrics[field] - row[old]); assert error < 1e-8
            max_metric_error = max(max_metric_error, error)
        daily_returns.append(nav[1:] / nav[:-1] - 1)
        proximity.append(dict(id=key, **metrics, monthly_target=monthly_target(result['daily']),
            **distance(metrics, reference['references']['primary'], near_fraction=reference['near_fraction']),
            candidate_validation_pending=True))
        report, found = inspect_account(raw, ix, result, key, row['ceiling'], caps)
        exposure.append(report); events.extend(found)
        relevant = [e for e in found if e['category'] in ['official_sector', 'small_cap']]
        screen = screen_account(relevant, result, raw, ix); screens.extend(screen)
        unresolved = [e for e in screen if e['status'] == 'still_above_no_relevant_sale']
        if not unresolved:
            continue
        panel = dict(raw)
        panel['sector_cap'] = np.minimum(raw['sector_cap'], .2) if row['ceiling'] == 'research20' else raw['sector_cap'].copy()
        cache_key = model, row['refresh']
        if cache_key not in held:
            matrix = np.load(source / 'scores' / (model + '.npy'), allow_pickle=False)
            held[cache_key] = snapshot_scores(matrix, raw['eligible'], ix['dates'], '20260101', '20260923', row['refresh'])
        alpha, origins = held[cache_key]
        reproduced_result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12,
            weight=.05, max_orders=row['max_orders'], rank_buffer=row['buffer'], rebalance=5,
            rebalance_band=.0005, return_trades=True, return_plans=True, planning_price='open',
            action_release_dates=release, stock_cap_schedule=caps, locked_repair=True)
        dates = {d: i for i, d in enumerate(ix['dates'])}
        for trade in reproduced_result['trades']:
            trade['portfolio_score_date'] = ix['dates'][origins[dates[trade['signal_date']]]]
        reproduced_result['metrics'].update(assess_weeks(reproduced_result))
        plans = reproduced_result['plans']; reproduced_result['plans'] = []
        assert reproduced_result == result
        reproduced.append(dict(id=key, all_fields_exact=True))
        classified = None
        for event in unresolved:
            detail = plan_checks(event, plans, panel, ix, raw['sector_cap'])
            detail.update(next_filled_orders=sum(t['date'] == event['next_date'] for t in result['trades']), max_orders=row['max_orders'])
            details.append(detail)
            for plan in detail['sell_plans']:
                if plan['execution_price_missing'] and plan['available_capacity'] == 0:
                    continue
                if classified is None:
                    classified = {(v['date'], v['code'], v['side']): v
                                  for v in classify_plans(panel, ix, plans, result['trades'], row['max_orders'])}
                record = classified[event['next_date'], plan['code'], 'sell']
                assert record['filled_qty'] == 0
                available.append(dict(id=key, ceiling=row['ceiling'], previous_date=event['date'],
                    category=event['category'], sector=event.get('sector'), max_orders=row['max_orders'], **record))
    unique = {}
    for r in available:
        key = tuple(r[k] for k in ['id', 'date', 'code', 'side'])
        assert key not in unique or unique[key] == r['stage']
        unique[key] = r['stage']
    unplanned = [e for e in details if not e['sell_plans'] and
                 (e['category'] == 'small_cap' or not e['prior_weight_within_next_limit'])]
    for name, sha in manifest.items():
        assert digest(source / name) == sha
    output.mkdir(parents=True)
    np.save(output / 'daily_returns.npy', np.column_stack(daily_returns))
    write(output / 'account_ids.json', [r['id'] for r in rows])
    write(output / 'summary.json', rows)
    write(output / 'paper_proximity.json', proximity)
    write(output / 'closing_exposure_accounts.json', exposure)
    write(output / 'closing_exposure_events.json', events)
    write(output / 'closing_followup.json', dict(summary=summarize(screens, details), screen=screens,
        unresolved_details=details, exact_replays=reproduced, available_repair_details=available,
        available_repair_unique_plans=len(unique), available_repair_stage_counts=dict(Counter(unique.values())),
        unplanned_exceptions=unplanned, full_contest_compliance_certified=False))
    hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    write(output / 'artifact_hashes.json', hashes)
    write(output / 'complete.json', dict(status='independent_case_audit_complete_main_review_pending',
        case=case, accounts=128, dates=expected_dates, exact_closing_replays=len(reproduced),
        maximum_nav_error=max(r['maximum_nav_error'] for r in exposure), maximum_metric_error=max_metric_error,
        unplanned_events=len(unplanned), available_unfilled_repair_plans=len(unique),
        source_account_manifest_sha256=digest(source / 'artifact_hashes.json'),
        source_manifest=manifest, market_manifest_sha256=digest(market_root / 'manifest.json'),
        validator_sha256=digest(Path(__file__)), family_inference_pending=True,
        independent_confirmation=False, full_contest_compliance_certified=False, orders_submitted=False))


def watch(market_root, sidecar):
    torch.set_num_threads(1)
    verify(market_root)
    reference = json.loads((sidecar / 'paper_registration.json').read_text())
    results = sidecar / 'results'; results.mkdir(exist_ok=True)
    done = []
    for p in sorted(results.glob('*/complete.json')):
        row = json.loads(p.read_text()); assert row['case'] == p.parent.name
        for name, sha in json.loads((p.parent / 'artifact_hashes.json').read_text()).items():
            assert digest(p.parent / name) == sha
        done.append(row['case'])
    active = {}
    (sidecar / 'logs').mkdir(exist_ok=True)
    while True:
        status_path = market_root / 'queue_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        for case in status.get('accounts_complete', []):
            if case in done or case in active: continue
            if len(active) >= 2: break
            log = (sidecar / 'logs' / (case + '.log')).open('x')
            proc = subprocess.Popen([sys.executable, '-m', 'quant.timefolio_heatmap_fleet_validation',
                '--market-root', str(market_root), '--sidecar', str(sidecar), '--case', case],
                stdout=log, stderr=subprocess.STDOUT)
            active[case] = proc, log
        for case, (proc, log) in list(active.items()):
            if proc.poll() is None: continue
            log.close(); del active[case]
            if proc.returncode or not (results / case / 'complete.json').exists():
                raise RuntimeError('Independent case audit failed: ' + case)
            done.append(case)
        finished = status.get('state') == 'complete' and set(done) == set(status['assigned_cases'])
        write(sidecar / 'status.json', dict(at=time.time(), state='complete' if finished else 'running',
            completed=done, running=list(active), cpu_parallel=2))
        if finished: return
        if status.get('failed'): raise RuntimeError('Parent queue failed; retain audit evidence')
        time.sleep(5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--market-root', type=Path, required=True)
    ap.add_argument('--sidecar', type=Path, required=True); ap.add_argument('--case'); args = ap.parse_args()
    try:
        if args.case:
            torch.set_num_threads(1)
            assert args.case in {c['id'] for c in cases()}
            reference = json.loads((args.sidecar / 'paper_registration.json').read_text())
            validate_case(args.market_root, args.market_root / 'accounts' / args.case,
                          args.sidecar / 'results' / args.case, reference)
        else:
            watch(args.market_root.resolve(), args.sidecar.resolve())
    except Exception as exc:
        name = 'failure_' + args.case + '.json' if args.case else 'failure.json'
        write(args.sidecar / name, dict(at=time.time(), error=repr(exc)))
        raise
