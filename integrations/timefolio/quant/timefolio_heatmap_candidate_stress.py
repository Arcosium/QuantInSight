"""Frozen-candidate execution stress checks; no fitting or parameter selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from quant.timefolio_heatmap_fleet_accounts import load_market
from quant.timefolio_heatmap_fleet_audit_helpers import inspect_account
from quant.timefolio_heatmap_fleet_metrics import monthly_target
from quant.timefolio_heatmap_gpu_worker import digest, verify, write
from quant.timefolio_heatmap_locked_replay import replay
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps


def run(root, request, output):
    torch.set_num_threads(1); verify(root)
    assert not output.exists(); output.mkdir(parents=True)
    raw, ix, release, _, _ = load_market(root)
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    results = []
    scenarios = [('baseline', .0005, .05), ('slippage10bp', .001, .05),
                 ('slippage20bp', .002, .05), ('participation2_5pct', .0005, .025),
                 ('participation1pct', .0005, .01)]
    for case, ids in request['cases'].items():
        source = root / 'accounts' / case
        for relative, sha in json.loads((source / 'artifact_hashes.json').read_text()).items():
            path = source / relative
            assert path.resolve().is_relative_to(source.resolve()) and digest(path) == sha
        rows = {r['id']: r for r in json.loads((source / 'summary.json').read_text())}
        for key in ids:
            row = rows[key]
            original = json.loads((source / 'portfolios' / (key + '.json')).read_text())
            panel = dict(raw)
            panel['sector_cap'] = np.minimum(raw['sector_cap'], .2) if row['ceiling'] == 'research20' else raw['sector_cap'].copy()
            scores = np.load(source / 'scores' / (row['model'] + '.npy'), allow_pickle=False)
            alpha, _ = snapshot_scores(scores, raw['eligible'], ix['dates'], '20260101', '20260923', row['refresh'])
            for name, slip, participation in scenarios:
                result = replay(panel, ix, alpha, '20260101', '20260923', top_n=12,
                    weight=.05, max_orders=row['max_orders'], rank_buffer=row['buffer'], rebalance=5,
                    rebalance_band=.0005, return_trades=True, return_plans=False, planning_price='open',
                    action_release_dates=release, stock_cap_schedule=caps, locked_repair=True,
                    slip=slip, participation=participation)
                if name == 'baseline':
                    assert result['daily'] == original['daily'], 'Frozen candidate daily NAV changed'
                account_path = output / (key + '__' + name + '.json'); write(account_path, result)
                audit, events = inspect_account(raw, ix, result, key, row['ceiling'], caps)
                results.append(dict(id=key, case=case, scenario=name, slip=slip, participation=participation,
                    monthly=monthly_target(result['daily']), independent_arithmetic=audit,
                    closing_events=len(events), account_sha256=digest(account_path)))
    write(output / 'results.json', results)
    write(output / 'complete.json', dict(status='execution_stress_complete_main_review_pending',
        cases=request['cases'], results=len(results), no_refitting=True, no_parameter_selection=True,
        heldout_accessed=False, multiple_testing_review_pending=True, full_contest_compliance_certified=False,
        results_sha256=digest(output / 'results.json'), source_sha256=digest(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.root, json.loads(args.request.read_text()), args.output)
