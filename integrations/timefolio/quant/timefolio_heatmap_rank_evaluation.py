"""All ranking forecasts under matched Timefolio limits and the retained family."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import DEST as BASE_ROOT, amend_actions, release_audit
from quant.timefolio_heatmap_concentration import DEST as PRIOR_ROOT, prior_family, comparator as prior_comparator
from quant.timefolio_heatmap_rank_training import DEST as TRAIN_ROOT, CONFIGS
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_dated_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT, AlternativeNet, predict
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_rank_evaluation_v1')
CASES = [dict(id=f'n{n}__orders{budget}__refresh{refresh}', top_n=n, max_orders=budget, refresh=refresh)
         for n in [4, 12] for budget in [3, 10] for refresh in [1, 5]]


def freeze():
    training = json.loads((TRAIN_ROOT/'protocol.json').read_text())
    base = json.loads((BASE_ROOT/'protocol.json').read_text())
    files = [Path(__file__), TRAIN_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json', BASE_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['rank_training', 'concentration', 'consensus', 'snapshot', 'stock_limits', 'week_boundaries',
               'dated_replay', 'action_amendment', 'execution_amendment', 'planned_audit', 'seed_evaluation',
               'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = {'training': training, 'base': base, 'cases': CASES,
            'policy': 'all6 newly trained forecasts plus online_nonimage; target5%, gross80%, sector min(statutory,20%), small-cap30%, execution-date stock caps; top4/12, 3/10fills/day, 1/5session score refresh, rebalance5, NAV5bp band',
            'turnover': 'completed last holiday week included; same verified KRX closures',
            'comparators': 'cash, same-case MLP with same horizon and training loss, and same-case original causal online_nonimage; no per-result comparator selection',
            'joint_family': '3128 retained full-period hypotheses plus3CNNs x8cases x3comparators =3200',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57, 'alpha': .025},
            'gate': 'positive net return, MDD>-20%, >=2positivequarters, fewer than4low-turnover weeks; adjustedp<0.025 and simultaneous lower95>0 for all3comparators/bothblocks; independent confirmation and seed/stress replication remain required',
            'paired_loss_effect': 'H5 pairwise minus H5 Huber results are descriptive only; same date batches isolate the loss within each architecture',
            'status': 'adaptive development only; prior unverified data/queue/action limitations retained; separate short news family remains separate',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Ranking evaluation protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files): shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    return spec


def prior_comparisons():
    family, differences, returns, _ = prior_family()
    prior = pd.read_csv(PRIOR_ROOT/'joint_bootstrap.csv')
    added = prior[prior.origin == 'concentration'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        model, case = key.split('__', 1)
        baseline = 0. if comp == 'cash' else returns(PRIOR_ROOT, prior_comparator(model)+'__'+case)
        family.append((key, comp, origin)); differences.append(returns(PRIOR_ROOT, key)-baseline)
    if len(family) != 3128 or len(set(family)) != 3128: raise AssertionError('Prior family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(np.mean(x)-means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Retained returns changed')
    return family, differences, returns, error


def validate_training(di, dates):
    rows = sorted((TRAIN_ROOT/'models').glob('*/*.json'))
    if len(rows) != 54: raise AssertionError('Ranking training is incomplete')
    records = []
    for path in rows:
        row = json.loads(path.read_text())
        if (row['last_refit_label'] >= row['first_execution']
                or row['last_inner_train_label'] >= row['first_inner_validation_signal']):
            raise AssertionError('Invalid training origin')
        if not path.with_suffix('.pt').exists() or not path.with_suffix('.pred.npy').exists():
            raise AssertionError('Missing training artifact')
        records.append({'model': row['config']['id'], 'month': row['fold']['month'], 'purge_valid': True})
    raw = np.load(MODEL_ROOT/'images_raw.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(raw[:, None], copy=True)); reproduced = []
    for cfg in CONFIGS:
        path = TRAIN_ROOT/'models'/cfg['id']/'202609.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config'] != cfg: raise AssertionError('Checkpoint configuration differs')
        model = AlternativeNet(cfg['architecture']); model.load_state_dict(saved['state_dict'])
        expected = np.load(path.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
        actual = predict(model, x, ids); error = float(np.max(np.abs(actual-expected[ids])))
        if error > 1e-6: raise AssertionError('Saved ranking forecasts failed reproduction')
        reproduced.append({'model': cfg['id'], 'rows': len(ids), 'maximum_error': error})
        del model
    atomic_json(DEST/'training_audit.json', {'origins': records, 'september_checkpoints': reproduced})
    files = sorted((TRAIN_ROOT/'models').glob('*/*.pred.npy'))
    atomic_json(DEST/'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def run():
    spec = freeze(); torch.set_num_threads(4); base = spec['base']; training = spec['training']
    for f, digest in training['hashes'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest() != digest: raise AssertionError('Training source changed')
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, base['actions'])
    caps = historical_stock_caps(ix['codes'], ix['dates']); validate_training(di, ix['dates'])
    old, _, _ = load_scores(MODEL_ROOT, p, ix, ci, di, base['prior']['model_protocol'])
    scores = {'online_nonimage': old['online_nonimage']}; del old
    for cfg in CONFIGS:
        values = merge_months(TRAIN_ROOT/'models'/cfg['id'], di, ix['dates'])
        scores[cfg['id']] = score_matrix(values, ci, di, p['close'].shape)
    p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    start, end = training['evaluation_start'], training['evaluation_end']
    date_index = {date: d for d, date in enumerate(ix['dates'])}
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True); rows = []; audits = {}
    for model, raw in scores.items():
        held = {r: snapshot_scores(raw, p['eligible'], ix['dates'], start, end, r) for r in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]; key = model+'__'+case['id']; path = folder/(key+'.json')
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(p, ix, alpha, start, end, top_n=case['top_n'], weight=.05,
                                max_orders=case['max_orders'], rebalance=5, rebalance_band=.0005,
                                return_trades=True, planning_price='open', action_release_dates=release,
                                stock_cap_schedule=caps)
                for trade in result['trades']:
                    origin = origins[date_index[trade['signal_date']]]
                    if origin < 0: raise AssertionError('Missing score origin')
                    trade['portfolio_score_date'] = ix['dates'][origin]
                result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
            check = audit_fills(p, ix, result)
            check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
            check['announced_action_errors'] = release_audit(p, ix, result, release)
            check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
            check['snapshot_origin_errors'] = []
            for trade in result['trades']:
                d = date_index[trade['signal_date']]; origin = origins[d]
                if not 0 <= origin <= d or trade['portfolio_score_date'] != ix['dates'][origin]:
                    check['snapshot_origin_errors'].append(trade['date'])
            if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['dated_hynix_audit']['post_buy_limit_errors'] or check['snapshot_origin_errors']
                    or check['maximum_nav_reconstruction_error_krw'] > .01):
                atomic_json(DEST/'failed_audit.json', {'id': key, 'audit': check}); raise AssertionError('Ranking replay failed audit')
            audits[key] = check; quarters = calendar_blocks(result['daily'])
            rows.append(dict(id=key, model=model, case=case['id'], **result['metrics'],
                             **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
        print(json.dumps({'model': model, 'portfolios': len(rows)}), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    atomic_json(DEST/'independent_audit.json', audits)
    family, differences, returns, error = prior_comparisons()
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses': len(family), 'maximum_mean_error': error})
    neural = frame[frame.model.str.startswith('cnn_')]
    for row in neural.to_dict('records'):
        control = row['model'].replace('cnn_', 'mlp_', 1)+'__'+row['case']
        for comp, baseline in [('cash', 0.), ('matched_mlp', returns(DEST, control)),
                               ('online_nonimage', returns(DEST, 'online_nonimage__'+row['case']))]:
            family.append((row['id'], comp, 'rank')); differences.append(returns(DEST, row['id'])-baseline)
    if len(family) != 3200: raise AssertionError('Ranking comparison family changed')
    statistics = []
    for block in spec['bootstrap']['blocks']:
        boot = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            statistics.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                   **{k: float(boot[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in neural.to_dict('records'):
        infer = stats[(stats.id == row['id']) & (stats.origin == 'rank')]
        if len(infer) != 6: raise AssertionError('Missing ranking comparisons')
        gate = row['return'] > 0 and row['mdd'] > -.20 and row['positive_blocks'] >= 2 and not row['four_week_turnover_stop']
        if gate and (infer.adjusted_p < .025).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json', {'status': 'development_only', 'portfolios': len(rows),
                                               'joint_hypotheses': len(family), 'candidate_gate_passed': candidates,
                                               'independent_confirmation': False})
    print(json.dumps({'complete': True, 'candidates': candidates}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()
