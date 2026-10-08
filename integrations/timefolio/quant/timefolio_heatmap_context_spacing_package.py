"""Register216 new fits with108 existing stride1 fits retained as controls."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from quant.timefolio_heatmap_gpu_worker import digest, verify, write
from quant.timefolio_heatmap_context_spacing import encode
from quant.timefolio_heatmap_context_spacing_models import cases, evaluation_cases
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, check_hashes
from quant.timefolio_heatmap_study import context

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'vault/ArcTrade/timefolio_heatmap'
ROOT = BASE / '20260929_context_spacing_lab_v1'
FEATURE = BASE / '20260929_feature_lab_v1'
FEATURE_EVAL = BASE / '20260929_feature_evaluation_v1'


def prepare(root):
    source = BASE / '20260929_geometry_lab_v2/package'; verify(source); verify(FEATURE / 'package')
    check_hashes(json.loads((DATA_ROOT / 'data_hashes.json').read_text()))
    review = json.loads((FEATURE_EVAL / 'validation_complete.json').read_text())
    assert review['status'] == 'numerical_validation_complete'
    check_hashes(review['main_review_evidence_hashes'])
    parent = json.loads((FEATURE / 'package/plan.json').read_text())
    panel, ix, ci, di = context(DATA_ROOT)
    assert len(ix['dates']) == 255 and ix['dates'][-1] == '20260923'
    if root.exists(): raise RuntimeError('Fresh context spacing root required')
    package = root / 'package'; package.mkdir(parents=True)
    raw = np.load(source / 'images_history.npy', mmap_mode='r')
    assert raw.shape == (46596, 32, 65)
    outputs = {f's{s}_{m}':np.lib.format.open_memmap(package/f'images_s{s}_{m}.npy', mode='w+', dtype='uint8',
               shape=(len(raw), 2, 32, 65)) for s in [1,2,4] for m in ['constant','available']}
    counts = {s:dict(samples_with_nonfinite=0, nonfinite_daily_pixels=0, changed_image_samples=0) for s in [1,2,4]}
    for start in range(0, len(raw), 256):
        end = min(start + 256, len(raw)); image = np.array(raw[start:end])
        for stride in [1,2,4]:
            observed = encode(image, panel, ci[start:end], di[start:end], stride, 'available')
            outputs[f's{stride}_available'][start:end] = observed
            counts[stride]['samples_with_nonfinite'] += int((observed[:,1] == 0).any(axis=(1,2)).sum())
            counts[stride]['nonfinite_daily_pixels'] += int((observed[:,1] == 0).sum())
            counts[stride]['changed_image_samples'] += int((observed[:,0] != image).any(axis=(1,2)).sum())
            observed[:,1] = 255
            outputs[f's{stride}_constant'][start:end] = observed
    for out in outputs.values(): out.flush()
    for mask in ['constant','available']:
        before = np.load(FEATURE / f'package/images_mask_{mask}.npy', mmap_mode='r')
        for start in range(0, len(raw), 256):
            np.testing.assert_array_equal(before[start:start+256], outputs[f's1_{mask}'][start:start+256])
    assert [counts[s]['samples_with_nonfinite'] for s in [1,2,4]] == [7476,10101,14637]
    assert [counts[s]['nonfinite_daily_pixels'] for s in [1,2,4]] == [51783*13,83421*13,138831*13]
    write(root/'encoding_audit.json', dict(samples=len(raw), stride1_feature_inputs_exact=True,
        missingness=counts, signal_cutoff=ix['dates'][-1], non_daily_rows_exact=True, latest13columns_exact=True,
        interpretation='Only15 daily-summary rows use wider date spacing. The17 intraday/time rows retain the original consecutive5days; column time scales therefore differ across these row groups.'))
    for name in ['labels_masks.npz','reference.py']: shutil.copyfile(source/name, package/name)
    files = {'engine.py':'timefolio_heatmap_gpu_worker.py','base_models.py':'timefolio_heatmap_lab_models.py',
        'feature_models.py':'timefolio_heatmap_feature_models.py','models.py':'timefolio_heatmap_context_spacing_models.py',
        'worker.py':'timefolio_heatmap_feature_worker.py','scheduler.py':'timefolio_heatmap_feature_queue.py'}
    for target, name in files.items(): shutil.copyfile(PROJECT/'quant'/name, package/target)
    configs = cases(); all_configs = evaluation_cases()
    jobs = [dict(id=f"{c['id']}_seed{s}", case=c['id'], seed=s) for s in [17,29,43] for c in configs]
    plan = dict(cases=configs, evaluation_cases=all_configs, seeds=[17,29,43], jobs=jobs, folds=parent['folds'],
        boundaries=parent['boundaries'], fits=216, reused_control_fits=108, source_period=parent['source_period'],
        selection=parent['selection'], fixed_account_grid=parent['fixed_account_grid'],
        representations=dict(strides=[1,2,4], daily_offsets=[[-4,-3,-2,-1,0],[-8,-6,-4,-2,0],[-16,-12,-8,-4,0]],
            intraday='Original consecutive5days in17rows; exact stored codes preserved.', daily='15 archived daily rows; no reindexing or interpolation of missing values.',
            masks='Both networks always have2channels. Constant255 versus archived daily-feature finiteness255/0; matched initialization.'),
        evaluation=dict(trained_streams=48, untrained_streams=48, online_streams=1, accounts=1552,
            new_accounts=1024, legacy_accounts=528, prior_hypotheses=64368, new_hypotheses=4352, joint_hypotheses=68720,
            members=['seed17','seed29','seed43','ensemble'], comparators=['cash','own_untrained','online','matched_other_architecture'],
            wider_context_also_same_mask_stride1=True, availability_also_same_stride_constant_mask=True,
            omit_self=True, formula_also_own_research20=True,
            bootstrap=dict(blocks=[5,10], draws=4000, seed=57, alpha=.025, implementation='streamed'),
            news_family=dict(hypotheses=147, alpha=.025, separate=True)),
        execution='Unchanged locked_repair=True ledger and16policies. Replayed stride1/online controls must exactly equal the reviewed feature accounts.',
        rationale='Compare5/9/17-session daily context spans with unchanged5-day intraday information, sample population, labels and model capacity.',
        interpretation='Adaptive development after feature, geometry and optimizer reviews. No capacity/epoch portfolio outcomes or new spacing outcomes used before registration. Reused controls are disclosed, not new independent trials.',
        gate=parent['gate'], limitations=parent['limitations'], export=parent['export'], billing=parent['billing'])
    write(package/'plan.json', plan)
    evidence = [Path(__file__), source/'manifest.json', FEATURE/'package/manifest.json', FEATURE_EVAL/'validation_complete.json',
        DATA_ROOT/'data_hashes.json', BASE/'20260929_temporal_spacing_readiness_v1/main_review.json',
        BASE/'20260929_streamed_market_family_check_v1/main_review.json', PROJECT/'quant/timefolio_heatmap_context_spacing.py',
        PROJECT/'quant/timefolio_heatmap_daily_history.py', PROJECT/'tests/test_timefolio_context_spacing.py']
    evidence += [PROJECT/'quant'/name for name in files.values()]
    write(root/'provenance.json', dict(hashes={str(p.resolve()):digest(p) for p in evidence},
        feature_performance_read=True, capacity_and_epoch_account_performance_read=False, own_prediction_results_read=False))
    manifest = {p.name:digest(p) for p in sorted(package.iterdir())}; write(package/'manifest.json', manifest)
    with tarfile.open(root/'package.tar.gz', 'w:gz', compresslevel=1) as stream:
        for p in sorted(package.iterdir()): stream.add(p, arcname=p.name)
    write(root/'export_receipt.json', dict(files=list(manifest), archive_sha256=digest(root/'package.tar.gz'),
        manifest_sha256=digest(package/'manifest.json'), fits=216, jobs=24, package_bytes=(root/'package.tar.gz').stat().st_size))
    print(dict(root=str(root), new_fits=216, reused_fits=108, jobs=24, accounts=1552, joint_hypotheses=68720), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
