"""One audited monthly CNN fit/refit job; no implicit broker or cloud actions.

Dataset manifest arrays: images (float32 [-1,1] N,1,H,T with H in8,16,22), returns,
signal_index, label_end_index, security_key, eligible, dates. Each array is an
individual .npy file with a SHA256 in the manifest. Image/eligibility rows must
have been constructed using only information available on the signal date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from quant.timefolio_cnn_core import RankCNN, date_loss, purged_masks


def groups(signal, mask, keys):
    selected = np.flatnonzero(mask)
    result = []
    for day in np.unique(signal[selected]):
        ids = selected[signal[selected] == day]
        ids = ids[np.argsort(keys[ids], kind='stable')]
        if len(np.unique(keys[ids])) != len(ids):
            raise ValueError('Duplicate security on a date')
        if len(ids) >= 2:
            result.append(ids)
    return result


def image_batch(images, ids, device):
    x = torch.as_tensor(np.array(images[ids], dtype=np.float32, copy=True), device=device)
    if not torch.isfinite(x).all() or x.min() < -1.00001 or x.max() > 1.00001:
        raise ValueError('Nonfinite or unscaled heatmap batch')
    return x


def predict(model, images, ids, *, device, batch_size=256):
    model.eval(); out = []
    with torch.no_grad():
        for lo in range(0, len(ids), batch_size):
            out.append(model(image_batch(images, ids[lo:lo+batch_size], device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def fit(images, returns, signal, keys, train, validation, config, *, device='cpu', epochs=None,
        account_validator=None):
    """Select epochs on a registered inner criterion, or refit a fixed count."""
    if account_validator is not None and (epochs is not None or not validation.any()):
        raise ValueError('Account epoch selection requires a separate inner validation split')
    if np.any(train & validation) or not np.isfinite(returns[train | validation]).all():
        raise ValueError('Overlapping splits or missing training labels')
    if account_validator is not None and train[account_validator.sample_ids].any():
        raise ValueError('Training rows cannot enter the inner account validation')
    train_groups, val_groups = groups(signal, train, keys), groups(signal, validation, keys)
    if not train_groups or (validation.any() and not val_groups):
        raise ValueError('Missing eligible date groups')
    if max(map(len, train_groups)) > config['max_date_group']:
        raise ValueError('Unregistered universe size; do not silently subsample a date')
    if min(map(len, train_groups)) < config['top_k']:
        raise ValueError('Insufficient members for registered top-k')
    torch.manual_seed(config['seed'])
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(config['seed'])
    model = RankCNN(config['width'], config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    count = config['epochs'] if epochs is None else epochs
    if count < 1:
        raise ValueError('Positive epoch budget required')
    history, best, best_state, best_epoch = [], None, None, 0
    for epoch in range(count):
        model.train(); losses = []
        for index in np.random.default_rng(config['seed'] + epoch).permutation(len(train_groups)):
            ids = train_groups[index]
            optimizer.zero_grad(set_to_none=True)
            scores = model(image_batch(images, ids, device))
            y = torch.as_tensor(np.array(returns[ids], dtype=np.float32, copy=True), device=device)
            loss = date_loss(scores, y, config['objective'], top_k=config['top_k'], temperature=config['temperature'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1, training_loss=float(np.mean(losses)))
        if account_validator is not None:
            from quant.timefolio_cnn_account_selection import selection_key
            scores = predict(model, images, account_validator.sample_ids, device=device)
            row['inner_account'] = account_validator.evaluate(scores)
            candidate = selection_key(row['inner_account'])
        else:
            metrics = []
            for ids in val_groups:
                if len(ids) < config['top_k']:
                    raise ValueError('Validation date has fewer than top-k members')
                scores = predict(model, images, ids, device=device)
                # ids already have stable security-key order for exact score ties.
                top = np.argsort(-scores, kind='stable')[:config['top_k']]
                metrics.append(float(returns[ids[top]].mean() - returns[ids].mean()))
            row['inner_topk_excess'] = float(np.mean(metrics)) if metrics else None
            candidate = (True, row['inner_topk_excess']) if metrics else None
        history.append(row)
        if candidate is None or best is None or candidate > best:
            best = candidate if candidate is not None else best
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    summary = dict(best_epoch=best_epoch, history=history,
        criterion='inner_account_net_profit' if account_validator is not None else 'inner_topk_excess',
        training_dates=len(train_groups), validation_dates=len(val_groups),
        training_rows=int(train.sum()), validation_rows=int(validation.sum()))
    if getattr(account_validator, 'provenance', {}).get('joint_epoch_and_policy_selection'):
        selected = history[best_epoch-1]['inner_account']
        summary.update(criterion='inner_account_net_profit_and_retention',
            selected_policy=dict(selected['selected_policy']),
            selected_policy_turnover_screen_passed=selected['turnover_screen_passed'])
    return model, summary


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def load_dataset(path, *, purpose='certified'):
    path = Path(path).resolve(); manifest = json.loads(path.read_text())
    if (path.parent/'RETIRED.json').exists():
        raise ValueError('Dataset retired after a data-quality audit')
    required = ['venue', 'session_calendar', 'corporate_actions', 'point_in_time_universe',
                'feature_availability', 'label_construction']
    readiness = manifest.get('readiness', {})
    if purpose == 'certified':
        if manifest.get('training_ready') is not True or any(readiness.get(k) is not True for k in required):
            raise ValueError('Dataset readiness audits incomplete for certified training')
    elif purpose == 'exploratory':
        if (manifest.get('exploratory_ready') is not True
                or manifest.get('contest_certified') is not False
                or not manifest.get('limitations')
                or any(readiness.get(k) is not True for k in ['feature_availability', 'label_construction'])):
            raise ValueError('Exploratory dataset needs explicit limitations and temporal audits')
    else:
        raise ValueError('Unknown training purpose')
    target_name = manifest.get('training_target', 'returns')
    if target_name not in ('returns', 'training_utility'):
        raise ValueError('Unregistered training target')
    names = ['images', 'returns', 'signal_index', 'label_end_index', 'security_key', 'eligible', 'dates']
    if target_name != 'returns': names.append(target_name)
    arrays = {}
    for name in names:
        spec = manifest['arrays'][name]; source = (path.parent / spec['path']).resolve()
        if not source.is_relative_to(path.parent) or sha(source) != spec['sha256']:
            raise ValueError('Dataset array outside package or hash mismatch')
        arrays[name] = np.load(source, mmap_mode='r', allow_pickle=False)
    n = len(arrays['images']); dates = arrays['dates']
    feature_rows = manifest.get('feature_rows', 8)
    if (feature_rows not in (8, 16, 22) or arrays['images'].ndim != 4
            or arrays['images'].shape[1:3] != (1, feature_rows)
            or arrays['images'].dtype != np.float32 or dates.ndim != 1
            or len(np.unique(dates)) != len(dates) or np.any(dates[1:] <= dates[:-1])
            or str(dates[-1]) > '20260923'
            or any(arrays[k].shape != (n,) for k in ['returns', 'signal_index', 'label_end_index', 'security_key', 'eligible'])
            or arrays['eligible'].dtype != bool
            or arrays[target_name].shape != (n,)
            or np.any(arrays['signal_index'] < 0) or np.any(arrays['signal_index'] >= len(dates))):
        raise ValueError('Invalid registered dataset geometry or reserved cutoff')
    return manifest, arrays


def run(args):
    from quant.timefolio_cnn_batch import policy_mode
    mode = policy_mode(vars(args), account_panel=args.account_panel)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic local CPU training fallback')
    if args.output.exists() or args.output.with_name(args.output.name + '.partial').exists():
        raise ValueError('Refusing to overwrite or resume an incomplete model job')
    manifest, a = load_dataset(args.dataset, purpose=args.purpose)
    fold = next(f for f in manifest['folds'] if f['id'] == args.fold)
    target_name = manifest.get('training_target', 'returns')
    s, end, y = a['signal_index'], a['label_end_index'], a[target_name]
    valid = a['eligible'] & np.isfinite(y)
    masks = purged_masks(s, end, valid, validation_start=fold['validation_start'],
        test_start=fold['test_start'], test_end=fold['test_end'])
    # Prediction eligibility uses signal-time metadata, never realized returns.
    prediction = masks['prediction'] & a['eligible']
    if not prediction.any():
        raise ValueError('No eligible predictions in registered outer fold')
    if len(np.unique(s[masks['train']])) < 126 or len(np.unique(s[masks['validation']])) < 20:
        raise ValueError('Insufficient purged training/validation history')
    cfg = dict(objective=args.objective, seed=args.seed, width=16, dropout=.1,
        epochs=6, lr=.0007, weight_decay=.001, top_k=10, temperature=.2, max_date_group=512)
    cfg.update(training_target=target_name, downside_penalty=manifest.get('downside_penalty'),
               feature_rows=manifest.get('feature_rows', 8))
    validator = None
    if args.account_panel is not None:
        if mode == 'retention_grid':
            from quant.timefolio_cnn_policy_selection import load_policy_validator
            validator = load_policy_validator(args.account_panel, args.dataset, a, fold)
        else:
            from quant.timefolio_cnn_account_selection import load_validator
            validator = load_validator(args.account_panel, args.dataset, a, fold,
                                      rebalance=args.rebalance, rank_buffer=args.rank_buffer)
        cfg['account_selection'] = validator.provenance
    started = time.monotonic()
    model, selection = fit(a['images'], y, s, a['security_key'], masks['train'], masks['validation'], cfg,
                           device=args.device, account_validator=validator)
    del model
    if mode == 'retention_grid':
        cfg['selected_policy'] = selection['selected_policy']
    model, refit = fit(a['images'], y, s, a['security_key'], masks['refit'], np.zeros(len(s), bool), cfg,
        device=args.device, epochs=selection['best_epoch'])
    ids = np.flatnonzero(prediction)
    scores = predict(model, a['images'], ids, device=args.device)
    if not np.isfinite(scores).all() or not len(scores):
        raise ValueError('Missing/nonfinite outer predictions')
    tmp = args.output.with_name(args.output.name + '.partial'); tmp.mkdir(parents=True)
    torch.save(dict(config=cfg, state_dict=model.cpu().state_dict()), tmp/'model.pt')
    np.save(tmp/'sample_ids.npy', ids); np.save(tmp/'scores.npy', scores)
    proof = dict(config=cfg, fold=fold, selection=selection, refit=refit,
        dataset_manifest_sha256=sha(args.dataset), prediction_count=len(ids),
        last_training_label_index=int(end[masks['refit']].max()),
        last_selection_label_index=int(end[masks['validation']].max()),
        last_selection_account_mark_index=validator.last_mark_index if validator is not None else None,
        first_prediction_signal_index=int(s[ids].min()), seconds=time.monotonic()-started,
        purpose=args.purpose, limitations=manifest.get('limitations', []),
        account_validation_complete=False, contest_certified=False,
        source_sha256={p.name:sha(p) for p in [Path(__file__), Path(__file__).with_name('timefolio_cnn_core.py')]},
        artifacts={p.name:sha(p) for p in tmp.iterdir()})
    if validator is not None:
        for name in ['timefolio_cnn_account_selection.py', 'timefolio_cnn_account.py',
                     'timefolio_heatmap_locked_weighted_replay.py', 'timefolio_heatmap_replay.py',
                     'timefolio_heatmap_retention_replay.py', 'timefolio_heatmap_locked_targets.py']:
            proof['source_sha256'][name] = sha(Path(__file__).with_name(name))
        if mode == 'retention_grid':
            for name in ['timefolio_cnn_policy_selection.py', 'timefolio_cnn_batch.py']:
                proof['source_sha256'][name] = sha(Path(__file__).with_name(name))
        proof['inner_account_validation_complete'] = True
    (tmp/'receipt.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    tmp.rename(args.output)
    print(json.dumps(dict(output=str(args.output), predictions=len(ids), seconds=proof['seconds'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fold', required=True)
    ap.add_argument('--objective', choices=['mse', 'bce', 'listnet', 'topk_ce'], default='listnet')
    ap.add_argument('--seed', type=int, choices=[17, 29, 43], default=17)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--purpose', choices=['certified', 'exploratory'], default='certified')
    ap.add_argument('--account-panel', type=Path)
    ap.add_argument('--rebalance', type=int, choices=[5, 20], default=5)
    ap.add_argument('--rank-buffer', type=int, choices=[0, 5], default=0)
    ap.add_argument('--policy-selection', choices=['fixed', 'retention_grid'], default='fixed')
    run(ap.parse_args())
