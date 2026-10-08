"""Matched H10 Huber controls for the completed date-balanced ranking study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_rank_training import DEST as TRAIN_PRIOR, CONFIGS as PRIOR_CONFIGS, train_daywise
from quant.timefolio_heatmap_rank_evaluation import DEST as PRIOR, CASES, prior_comparisons
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_dated_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import (
    SOURCE, DEST as MODEL_ROOT, AlternativeNet, continuous_targets, monthly_folds, fold_masks, predict,
)
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_h10_control_v1')
CONFIGS = [dict(c, id=c['architecture']+'_huber_h10', objective='huber')
           for c in PRIOR_CONFIGS if c['objective'] == 'pairwise' and c['horizon'] == 10]


def freeze():
    training = json.loads((TRAIN_PRIOR/'protocol.json').read_text())
    prior = json.loads((PRIOR/'protocol.json').read_text())
    for path, digest in training['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest: raise AssertionError('Original training inputs changed')
    files = [Path(__file__), TRAIN_PRIOR/'protocol.json', PRIOR/'protocol.json', PRIOR/'joint_bootstrap.csv',
             TRAIN_PRIOR/'labels_h10.npy', SOURCE/'panel.npz', SOURCE/'panel_index.json', SOURCE/'samples.npz',
             MODEL_ROOT/'images_raw.npy']
    files += [p for c in CONFIGS for p in sorted((TRAIN_PRIOR/'models'/c['id'].replace('_huber_', '_pairwise_')).glob('*.json'))]
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['rank_training', 'rank_evaluation', 'concentration', 'consensus', 'snapshot', 'stock_limits',
               'week_boundaries', 'dated_replay', 'action_amendment', 'execution_amendment', 'planned_audit',
               'seed_evaluation', 'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = {'configs': CONFIGS, 'training_parent': training, 'evaluation_parent': prior, 'cases': CASES,
            'label_source': str(TRAIN_PRIOR/'labels_h10.npy'),
            'matching': 'H10 Huber differs from prior H10 pairwise only in loss and model id; raw images, labels, network, date batches, seed17, optimizer, validation metric, purge and evaluation conditions unchanged',
            'training': '2architectures x9monthly origins; reuse frozen train_daywise Huber implementation; independently recomputed labels must match prior labels; every fold must match prior training/validation/refit counts and date boundaries',
            'evaluation': 'same8cases; all16 new portfolios; dated stock limits and corrected completed-week turnover',
            'family': '3200 prior full-period hypotheses plus1CNN x8cases x3comparators =3224; cash, same-case H10 HuberMLP and original online_nonimage',
            'objective_effect': 'H10 pairwise versus H10 Huber for each architecture is descriptive; do not select the loss using an uncorrected incremental p-value',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57, 'alpha': .025},
            'gate': 'same positive net return, >=2positivequarters, MDD>-20%, fewer than4turnover failures; all three comparators/both blocks adjustedp<0.025 and simultaneous lower95>0; further seed/stress and independent confirmation still needed',
            'interpretation': 'adaptive development after observed ranking results; original corporate/book/dividend/universe limitations remain; reserved new session outcomes are not read',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('H10 control specification changed')
    if not path.exists():
        atomic_json(path, spec); frozen = DEST/'frozen_source'; frozen.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix in ['.py', '.json']: shutil.copy2(p, frozen/(f'{i:02d}_'+p.name))
    return spec


def train():
    spec = freeze(); torch.set_num_threads(4)
    p, ix, ci, di = context(SOURCE); p, _ = amend_actions(p, ix, spec['training_parent']['actions'])
    p['factor'] = np.cumprod(p['split'], axis=1)
    y = continuous_targets(p, ci, di, 10, 'sector')
    if not np.array_equal(y, np.load(spec['label_source']), equal_nan=True): raise AssertionError('H10 labels differ')
    dates = np.asarray(ix['dates']); valid = di+1 < len(dates); allowed = np.zeros(len(ci), bool)
    allowed[valid] = p['trade_allowed'][ci[valid], di[valid]+1].astype(bool)
    x = torch.from_numpy(np.array(np.load(MODEL_ROOT/'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    matches = []
    for cfg in CONFIGS:
        folder = DEST/'models'/cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder/(fold['month']+'.json')
            reference = json.loads((TRAIN_PRIOR/'models'/cfg['id'].replace('_huber_', '_pairwise_')/path.name).read_text())
            if {k:v for k,v in reference['config'].items() if k not in ['id','objective']} != {k:v for k,v in cfg.items() if k not in ['id','objective']}:
                raise AssertionError('Unmatched H10 model configuration')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if (int(tr.sum()) != reference['fit']['training_rows'] or int(va.sum()) != reference['fit']['validation_rows']
                    or int(rf.sum()) != reference['refit_rows']): raise AssertionError('Unmatched training data')
            matches.append({'model': cfg['id'], 'month': fold['month'], 'same_sample_counts_and_config_except_loss': True})
            if path.exists():
                if not path.with_suffix('.pt').exists() or not path.with_suffix('.pred.npy').exists(): raise RuntimeError('Incomplete H10 checkpoint')
                continue
            started = time.monotonic()
            model, fit = train_daywise(x, y, tr, va, cfg, di); del model
            model, _ = train_daywise(x, y, rf, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            values = predict(model, x, np.flatnonzero(pr))
            if not np.isfinite(values).all(): raise AssertionError('Non-finite H10 scores')
            out = np.full(len(ci), np.nan, np.float32); out[pr] = values
            torch.save({'config': cfg, 'state_dict': model.state_dict()}, path.with_suffix('.pt'))
            np.save(path.with_suffix('.pred.npy'), out)
            row = {'config': cfg, 'fold': fold, 'fit': fit, 'refit_rows': int(rf.sum()),
                   'last_inner_train_label': str(max(dates[di[tr]+10])),
                   'first_inner_validation_signal': str(min(dates[di[va]])),
                   'last_refit_label': str(max(dates[di[rf]+10])), 'first_execution': ix['dates'][fold['start']],
                   'last_execution': ix['dates'][fold['end']-1], 'seconds': round(time.monotonic()-started, 2)}
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != reference[key]: raise AssertionError('Unmatched fold boundary')
            atomic_json(path, row); del model
            print(json.dumps({'model': cfg['id'], 'month': fold['month'], 'epochs': fit['best_epoch'], 'seconds': row['seconds']}), flush=True)
    atomic_json(DEST/'training_matching.json', matches)
    print(json.dumps({'training_complete': True, 'fits': len(matches)}), flush=True)


def retained_family():
    family, differences, returns, _ = prior_comparisons()
    old = pd.read_csv(PRIOR/'joint_bootstrap.csv')
    for key, comp, origin in old[old.origin == 'rank'][['id','comparator','origin']].drop_duplicates().itertuples(index=False, name=None):
        model, case = key.split('__', 1)
        control = model.replace('cnn_', 'mlp_', 1)+'__'+case if comp == 'matched_mlp' else 'online_nonimage__'+case
        baseline = 0. if comp == 'cash' else returns(PRIOR, control)
        family.append((key, comp, origin)); differences.append(returns(PRIOR, key)-baseline)
    if len(family) != 3200 or len(set(family)) != 3200: raise AssertionError('Incomplete retained H10 family')
    means = old[old.block == 5].set_index(['id','comparator','origin'])['mean']
    error = max(abs(float(np.mean(x))-means.loc[key]) for key,x in zip(family,differences))
    if error > 1e-14: raise AssertionError('Retained means changed')
    return family, differences, returns, error


def evaluate():
    spec = freeze(); torch.set_num_threads(4)
    files = sorted((DEST/'models').glob('*/*.pred.npy'))
    if len(files) != 18: raise AssertionError('H10 training incomplete')
    atomic_json(DEST/'forecast_hashes.json', {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, spec['training_parent']['actions'])
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    date_index = {d:i for i,d in enumerate(ix['dates'])}; rows = []; audits = {}; checks = []
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True)
    raw_images = np.load(MODEL_ROOT/'images_raw.npy', mmap_mode='r')
    for cfg in CONFIGS:
        checkpoint = DEST/'models'/cfg['id']/'202609.pt'; saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['config'] != cfg: raise AssertionError('Saved configuration changed')
        net = AlternativeNet(cfg['architecture']); net.load_state_dict(saved['state_dict'])
        expected = np.load(checkpoint.with_suffix('.pred.npy')); ids = np.flatnonzero(np.isfinite(expected))
        actual = predict(net, torch.from_numpy(np.array(raw_images[:, None], copy=True)), ids); del net
        error = float(np.max(np.abs(actual-expected[ids])))
        if error > 1e-6: raise AssertionError('Checkpoint predictions differ')
        checks.append({'model': cfg['id'], 'rows': len(ids), 'maximum_error': error})
        matrix = score_matrix(merge_months(DEST/'models'/cfg['id'], di, ix['dates']), ci, di, p['close'].shape)
        for case in CASES:
            alpha, origins = snapshot_scores(matrix, p['eligible'], ix['dates'], '20260101', '20260923', case['refresh'])
            key = cfg['id']+'__'+case['id']; path = folder/(key+'.json')
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(p, ix, alpha, '20260101', '20260923', top_n=case['top_n'], weight=.05,
                                max_orders=case['max_orders'], rebalance=5, rebalance_band=.0005, return_trades=True,
                                planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
                for t in result['trades']:
                    d = date_index[t['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d: raise AssertionError('Invalid score origin')
                    t['portfolio_score_date'] = ix['dates'][origin]
                result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
            audit = audit_fills(p, ix, result); audit.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
            audit['announced_action_errors'] = release_audit(p, ix, result, release)
            audit['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
            audit['snapshot_origin_errors'] = []
            for t in result['trades']:
                d = date_index[t['signal_date']]; origin = origins[d]
                if not 0 <= origin <= d or t['portfolio_score_date'] != ix['dates'][origin]: audit['snapshot_origin_errors'].append(t['date'])
            if (audit['post_buy_limit_violations'] or audit['additional_errors'] or audit['announced_action_errors']
                    or audit['dated_hynix_audit']['post_buy_limit_errors'] or audit['snapshot_origin_errors']
                    or audit['maximum_nav_reconstruction_error_krw'] > .01):
                atomic_json(DEST/'failed_audit.json', {'id':key,'audit':audit}); raise AssertionError('H10 replay audit failed')
            audits[key] = audit; blocks = calendar_blocks(result['daily'])
            rows.append(dict(id=key, model=cfg['id'], case=case['id'], **result['metrics'],
                             **blocks, positive_blocks=sum(v > 0 for v in blocks.values())))
        print(json.dumps({'model':cfg['id'], 'portfolios':len(rows)}), flush=True)
    atomic_json(DEST/'checkpoint_reproduction.json', checks); atomic_json(DEST/'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    family, differences, returns, error = retained_family()
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses':len(family), 'maximum_mean_error':error})
    for row in frame[frame.model.str.startswith('cnn_')].to_dict('records'):
        for comp, baseline in [('cash',0.), ('matched_mlp',returns(DEST,'mlp_huber_h10__'+row['case'])),
                               ('online_nonimage',returns(PRIOR,'online_nonimage__'+row['case']))]:
            family.append((row['id'],comp,'h10_control')); differences.append(returns(DEST,row['id'])-baseline)
    if len(family) != 3224: raise AssertionError('H10 family changed')
    statistics = []
    for block in [5,10]:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i,(key,comp,origin) in enumerate(family):
            statistics.append(dict(id=key,comparator=comp,origin=origin,block=block,**{k:float(result[k][i]) for k in
                                   ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in frame[frame.model.str.startswith('cnn_')].to_dict('records'):
        infer = stats[(stats.id==row['id'])&(stats.origin=='h10_control')]
        if len(infer) != 6: raise AssertionError('Incomplete H10 comparisons')
        gate = row['return']>0 and row['mdd']>-.20 and row['positive_blocks']>=2 and not row['four_week_turnover_stop']
        if gate and (infer.adjusted_p<.025).all() and (infer.simultaneous_lower95>0).all(): candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json', {'status':'development_only','portfolios':len(rows),
                                               'joint_hypotheses':len(family),'candidate_gate_passed':candidates,
                                               'independent_confirmation':False})
    print(json.dumps({'complete':True,'candidates':candidates}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action',choices=['freeze','train','evaluate','all']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    if args.action in ['train','all']: train()
    if args.action in ['evaluate','all']: evaluate()
