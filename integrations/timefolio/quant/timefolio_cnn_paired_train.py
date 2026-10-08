"""One selection training trajectory, two independently selected account arms.

Only epoch selection differs within each seed/target pair. Identical selected
epoch counts reuse one deterministic refit and identical forecast values.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import json
import time
import numpy as np
import torch

from quant.timefolio_cnn_core import purged_masks
from quant.timefolio_cnn_train import fit, predict, load_dataset, sha
from quant.timefolio_cnn_account_selection import selection_key
from quant.timefolio_cnn_execution_selection import load_paired_validator, contest_policy


def arm_selections(shared):
    selections = {}
    for arm in ['legacy', 'contest']:
        history = []
        for row in shared['history']:
            account = copy.deepcopy(row['inner_account'])
            legacy = account.pop('paired_legacy_account')
            history.append(dict(epoch=row['epoch'], training_loss=row['training_loss'],
                                inner_account=legacy if arm == 'legacy' else account))
        # max preserves the first occurrence, so exact profit ties use fewer epochs.
        chosen = max(history, key=lambda row: selection_key(row['inner_account']))
        selections[arm] = dict(**{k:v for k,v in shared.items() if k not in ['history','best_epoch','criterion']},
                              history=history, best_epoch=chosen['epoch'], criterion='inner_account_net_profit', arm=arm)
    assert selections['contest']['best_epoch'] == shared['best_epoch']
    return selections


def run(args):
    if args.account_panel is None or args.rebalance != 5 or args.rank_buffer != 5:
        raise ValueError('The paired legacy arm requires registered R5/buffer5 and a panel')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no local training fallback')
    tmp = args.output.with_name(args.output.name+'.partial')
    if args.output.exists() or tmp.exists():raise ValueError('Refusing duplicate or partial job overwrite')
    manifest, a = load_dataset(args.dataset, purpose=args.purpose)
    if manifest.get('paired_account_policy') != contest_policy():
        raise ValueError('Dataset must bind the exact preregistered contest policy')
    fold = next(f for f in manifest['folds'] if f['id'] == args.fold)
    target = manifest.get('training_target', 'returns')
    s, end, y = a['signal_index'], a['label_end_index'], a[target]
    valid = a['eligible'] & np.isfinite(y)
    masks = purged_masks(s, end, valid, validation_start=fold['validation_start'],
                         test_start=fold['test_start'], test_end=fold['test_end'])
    prediction = masks['prediction'] & a['eligible']
    if (not prediction.any() or len(np.unique(s[masks['train']])) < 126
            or len(np.unique(s[masks['validation']])) < 20):
        raise ValueError('Insufficient purged fold coverage')
    cfg = dict(objective=args.objective, seed=args.seed, width=16, dropout=.1,
               epochs=6, lr=.0007, weight_decay=.001, top_k=10, temperature=.2, max_date_group=512,
               training_target=target, downside_penalty=manifest.get('downside_penalty'),
               feature_rows=manifest.get('feature_rows', 8))
    validator = load_paired_validator(args.account_panel, args.dataset, a, fold)
    cfg['account_selection'] = validator.provenance
    began = time.monotonic()
    selected_model, shared = fit(a['images'], y, s, a['security_key'], masks['train'], masks['validation'],
                                 cfg, device=args.device, account_validator=validator)
    del selected_model
    choices = arm_selections(shared);ids = np.flatnonzero(prediction);tmp.mkdir(parents=True)
    np.save(tmp/'sample_ids.npy', ids);refits = {};arms = {}
    for epochs in sorted(set(c['best_epoch'] for c in choices.values())):
        model, refit = fit(a['images'], y, s, a['security_key'], masks['refit'], np.zeros(len(s), bool),
                           cfg, device=args.device, epochs=epochs)
        scores = predict(model, a['images'], ids, device=args.device)
        if not len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid outer forecasts')
        refits[str(epochs)] = refit
        state = model.cpu().state_dict()
        for arm in [k for k,v in choices.items() if v['best_epoch'] == epochs]:
            arm_cfg = dict(cfg, selection_arm=arm)
            torch.save(dict(config=arm_cfg, state_dict=state), tmp/f'model_{arm}.pt')
            np.save(tmp/f'scores_{arm}.npy', scores)
            arms[arm] = dict(config=arm_cfg, selection=choices[arm], refit=refit,
                             model_file=f'model_{arm}.pt', score_file=f'scores_{arm}.npy')
        del model, state
    if choices['legacy']['best_epoch'] == choices['contest']['best_epoch']:
        assert (tmp/'scores_legacy.npy').read_bytes() == (tmp/'scores_contest.npy').read_bytes()
    names = ['timefolio_cnn_paired_train.py', 'timefolio_cnn_execution_selection.py',
             'timefolio_cnn_train.py', 'timefolio_cnn_core.py', 'timefolio_cnn_account_selection.py',
             'timefolio_cnn_account.py', 'timefolio_cnn_dataset.py', 'timefolio_cnn_batch.py',
             'timefolio_heatmap_locked_weighted_replay.py', 'timefolio_heatmap_replay.py',
             'timefolio_heatmap_retention_replay.py', 'timefolio_heatmap_locked_targets.py',
             'timefolio_cnn_core_retention_replay.py', 'timefolio_cnn_activity.py']
    proof = dict(arms=arms, fold=fold, dataset_manifest_sha256=sha(args.dataset),
                 shared_selection=shared, physical_selection_fits=1, physical_refits=len(refits), logical_arm_models=2,
                 shared_epoch_refit_reused=len(refits)==1, prediction_count=len(ids),
                 last_training_label_index=int(end[masks['refit']].max()),
                 last_selection_label_index=int(end[masks['validation']].max()),
                 last_selection_account_mark_index=validator.last_mark_index,
                 first_prediction_signal_index=int(s[ids].min()), seconds=time.monotonic()-began,
                 purpose=args.purpose, limitations=manifest.get('limitations', []),
                 inner_account_validation_complete=True, account_validation_complete=False, contest_certified=False,
                 source_sha256={name:sha(Path(__file__).with_name(name)) for name in names},
                 artifacts={p.name:sha(p) for p in tmp.iterdir()})
    (tmp/'receipt.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    tmp.rename(args.output)
    print(json.dumps(dict(output=str(args.output), predictions=len(ids), physical_refits=len(refits), selected_epochs={k:v['best_epoch'] for k,v in choices.items()})),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fold',required=True);ap.add_argument('--objective',choices=['mse','bce','listnet','topk_ce'],default='listnet')
    ap.add_argument('--seed',type=int,choices=[17,29,43],default=17);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--purpose',choices=['certified','exploratory'],default='certified')
    ap.add_argument('--account-panel',type=Path,required=True);ap.add_argument('--rebalance',type=int,choices=[5],default=5)
    ap.add_argument('--rank-buffer',type=int,choices=[5],default=5)
    run(ap.parse_args())
