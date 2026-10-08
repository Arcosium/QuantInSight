"""Build an explicit market-tensor export with a frozen 162-fit experiment."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from quant.timefolio_heatmap_gpu_worker import digest, write

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'vault/ArcTrade/timefolio_heatmap'
ROOT = BASE / '20260929_daily_history_gpu_v1'


def reference_source():
    """Copy exact definitions, avoiding credentials and local-path imports."""
    selections = {'study': ['BASE', 'HeatCNN'], 'walkforward': ['daily_ic', 'AlternativeNet'],
                  'rank_training': ['pairwise_loss', 'date_groups']}
    parts = ['import numpy as np\nimport torch\nfrom torch import nn\nfrom scipy.stats import spearmanr\n']
    for module, names in selections.items():
        source = (PROJECT / 'quant' / ('timefolio_heatmap_' + module + '.py')).read_text()
        found = {}
        for node in ast.parse(source).body:
            name = getattr(node, 'name', None)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name): name = node.targets[0].id
            if name in names: found[name] = ast.get_source_segment(source, node)
        assert set(found) == set(names)
        parts.extend(found[name] for name in names)
    return '\n\n'.join(parts) + '\n'


def prepare(root=ROOT):
    from quant.timefolio_heatmap_absolute_training import inputs, DEST as SNAPSHOT
    from quant.timefolio_heatmap_corrected_features import DEST as DATA
    from quant.timefolio_heatmap_walkforward import fold_masks, monthly_folds
    from quant.timefolio_heatmap_stock_relation_training import check_hashes
    prior = BASE / '20260929_sector_ceiling_v1'
    reviewed = json.loads((prior / 'validation_complete.json').read_text())
    assert reviewed['status'] == 'numerical_validation_complete' and not reviewed['candidates']
    history = BASE / '20260929_daily_history_features_v1'
    assert json.loads((history / 'main_review.json').read_text())['status'] == 'main_reviewed_feature_preparation'
    evidence = [prior / 'validation_complete.json', history / 'main_review.json']
    for manifest in [DATA / 'data_hashes.json', history / 'image_hashes.json']:
        check_hashes(json.loads(manifest.read_text())); evidence.append(manifest)
    root = Path(root)
    if root.exists(): raise RuntimeError('Use a fresh GPU study root')
    root.mkdir(); package = root / 'package'; package.mkdir()
    _, ix, _, di, y, allowed = inputs(); folds = monthly_folds(ix['dates'])
    assert len(ix['dates']) == 255 and ix['dates'][0] == '20250908' and ix['dates'][-1] == '20260923'
    assert np.array_equal(y, np.load(SNAPSHOT / 'labels_h10_absolute.npy'), equal_nan=True)
    arrays, boundaries = dict(y=y, di=di), {}
    for fold in folds:
        month = fold['month']; tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
        reference = SNAPSHOT / 'models/cnn_absolute_rank_h10_seed17' / (month + '.json')
        old = json.loads(reference.read_text()); evidence.append(reference)
        assert (int(tr.sum()), int(va.sum()), int(rf.sum())) == (old['fit']['training_rows'], old['fit']['validation_rows'], old['refit_rows'])
        arrays.update({month + '_' + k: v for k, v in dict(tr=tr, va=va, rf=rf, pr=pr).items()})
        boundaries[month] = {k:old[k] for k in ['last_inner_train_label', 'first_inner_validation_signal',
            'last_refit_label', 'first_execution', 'last_execution']}
        dates = np.asarray(ix['dates'])
        assert max(dates[di[tr] + 10]) < min(dates[di[va]])
        assert max(dates[di[rf] + 10]) < ix['dates'][fold['start']]
    assert len(folds) == 9
    np.savez(package / 'labels_masks.npz', **arrays)
    sources = {'snapshot':DATA / 'images_raw.npy', 'history':history / 'images_history.npy',
               'reverse':history / 'images_history_reverse.npy'}
    for encoding, source in sources.items():
        raw = np.load(source, mmap_mode='r'); assert raw.shape == (46596, 32, 65) and raw.dtype == np.uint8
        shutil.copyfile(source, package / ('images_' + encoding + '.npy')); evidence.append(source)
    (package / 'reference.py').write_text(reference_source())
    worker = PROJECT / 'quant/timefolio_heatmap_gpu_worker.py'
    shutil.copyfile(worker, package / 'worker.py')
    plan = dict(seeds=[17,29,43], architectures=['cnn','mlp'], encodings=['snapshot','history','reverse'],
        folds=folds, boundaries=boundaries, fits=162, fits_per_gpu=54, source_period=['20250908','20260923'],
        partition='One seed per GPU; all three encodings and both architectures on the same device. Retrain snapshot controls on GPU.',
        selection='Past20-session purged daily IC, max6epochs; restart/refit. No cross-fold or device winner selection.',
        evaluation='Future local evaluation must retain all12416 prior comparisons and all new registered accounts; separate147news family. This package only trains, no performance claim.',
        fixed_account_grid=dict(sector=['research20','formula'], top_n=12, weight=.05, gross=.8,
            orders=[3,10], refresh=[1,5], buffers=[12,24], band=.0005, rebalance=5),
        gate='Ensemble and all3seeds positive return,MDD>-20%,>=2positive quarters,<4failedturnoverweeks. Compare every trainedCNN with cash,matchedMLP,matcheduntrainedCNN,online and other2encodings; formula also ownresearch20. Append all hypotheses and require bothblock5/10 adjustedp<.025,lower95>0. Stress and fresh confirmation required.',
        limitations='Repeated255-session development, static sector/universe, approximate contest execution; GPU and CPU results not bitwise equivalent.',
        export='Only normalized market images, absolute labels, date indices, purged boolean masks, source and plan. No account records, news documents, keys or live DB.',
        billing=dict(max_pods=3,max_hourly_per_pod=.4,max_hours=2,max_base_estimate_usd=2.4,auto_recharge=False))
    write(package / 'plan.json', plan)
    evidence += [worker, Path(__file__), PROJECT / 'tests/test_timefolio_gpu_worker.py']
    write(root / 'provenance.json', dict(hashes={str(p):digest(p) for p in evidence},
        pending_cpu_history_plan_superseded=True, inference_scope='GPU cohort must be evaluated separately from old CPU cohort'))
    manifest = {p.name:digest(p) for p in sorted(package.iterdir())}
    write(package / 'manifest.json', manifest)
    with tarfile.open(root / 'package.tar.gz', 'w:gz', compresslevel=1) as archive:
        for p in sorted(package.iterdir()): archive.add(p, arcname=p.name)
    write(root / 'export_receipt.json', dict(files=list(manifest), package_bytes=(root/'package.tar.gz').stat().st_size,
        archive_sha256=digest(root/'package.tar.gz'), manifest_sha256=digest(package/'manifest.json'), fits=162))
    print(json.dumps(dict(root=str(root), fits=162, package_bytes=(root/'package.tar.gz').stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
