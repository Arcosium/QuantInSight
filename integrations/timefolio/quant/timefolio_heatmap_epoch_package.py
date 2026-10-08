"""Register270 model-month fits isolating epoch selection from fixed duration."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from quant.timefolio_heatmap_gpu_worker import digest, verify, write
from quant.timefolio_heatmap_epoch_lab import cases
from quant.timefolio_heatmap_stock_relation_training import check_hashes

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'vault/ArcTrade/timefolio_heatmap'
ROOT = BASE / '20260929_epoch_lab_v1'


def prepare(root):
    source = BASE / '20260929_geometry_lab_v2/package'; verify(source)
    parent = json.loads((source / 'plan.json').read_text())
    review = BASE / '20260929_training_selection_diagnostic_v1/main_review.json'
    diagnosis = json.loads(review.read_text()); check_hashes(diagnosis['hashes'])
    assert diagnosis['status'] == 'descriptive_training_selection_main_reviewed'
    assert not root.exists(); package = root / 'package'; package.mkdir(parents=True)
    for name in ['images_history.npy', 'labels_masks.npz', 'reference.py']:
        shutil.copyfile(source / name, package / name)
    files = {'engine.py':'timefolio_heatmap_gpu_worker.py', 'models.py':'timefolio_heatmap_lab_models.py',
        'training.py':'timefolio_heatmap_epoch_lab.py', 'worker.py':'timefolio_heatmap_epoch_worker.py',
        'scheduler.py':'timefolio_heatmap_lab_queue.py'}
    for target, name in files.items(): shutil.copyfile(PROJECT / 'quant' / name, package / target)
    configs = cases()
    order = [kind+'_'+mode for mode in ['fixed12','selected6','fixed6','fixed3','fixed1'] for kind in ['cnn','mlp']]
    jobs = [dict(id=f'{name}_seed{s}', case=name, seed=s) for s in [17,29,43] for name in order]
    assert len(jobs) == 30 and set(order) == {c['id'] for c in configs}
    plan = dict(cases=configs, seeds=[17,29,43], jobs=jobs, folds=parent['folds'], boundaries=parent['boundaries'],
        fits=270, source_period=parent['source_period'], fixed_account_grid=parent['fixed_account_grid'],
        selection='Inner-IC control selects among6epochs then restarts full purged refit. Fixed1/3/6/12 controls directly fit the same refit rows for the registered duration and never use the inner validation metric.',
        evaluation=dict(trained_streams=40, untrained_streams=8, online_streams=1, accounts=784,
            new_accounts=784, legacy_accounts=0, prior_hypotheses=60976, new_hypotheses=3392, joint_hypotheses=64368,
            members=['seed17','seed29','seed43','ensemble'],
            comparators=['cash','own_architecture_untrained','online','matched_other_architecture','same_architecture_selected6'],
            omit_self=True, formula_also_own_research20=True,
            bootstrap=dict(blocks=[5,10], draws=4000, seed=57, alpha=.025), news_family=dict(hypotheses=147,alpha=.025,separate=True)),
        execution='All accounts recomputed under locked_repair=True, unchanged16policy grid.',
        untrained='Every job saves untrained artifacts; require exact predictions and states across epoch rules within each architecture/seed. Eight canonical streams including ensembles.',
        regression='Selected6 controls must exactly reproduce geometry history_base/history_mlp artifacts. Any fixed refit whose duration equals that selected control must reproduce its corresponding monthly state and predictions exactly.',
        rationale='Training loss fell in all1296 reviewed fits, but baseline epoch selection uses only10validation dates. The comparison tests fixed durations; no claim that selection caused past performance.',
        information='Same46596 history inputs,H10labels,purged folds,pairwise loss,AdamW lr0.0007 and decay0.001.',
        interpretation='Prior geometry and optimizer performance and their training histories informed this design. Pending feature and capacity account outcomes were not used. Every fixed duration and seed is retained.',
        gate=parent['gate']+' Numerical paper proximity does not override the registered gates.',
        limitations=parent['limitations'], export=parent['export'], billing=parent['billing'])
    write(package / 'plan.json', plan)
    evidence = [Path(__file__), source/'manifest.json', review, PROJECT/'tests/test_timefolio_epoch_lab.py']
    evidence += [PROJECT/'quant'/name for name in files.values()]
    write(root/'provenance.json', dict(hashes={str(p.resolve()):digest(p) for p in evidence},
        prior_performance_and_training_histories_read=True, pending_feature_and_capacity_account_outcomes_read=False,
        own_prediction_results_read=False))
    manifest = {p.name:digest(p) for p in sorted(package.iterdir())}; write(package/'manifest.json',manifest)
    with tarfile.open(root/'package.tar.gz','w:gz',compresslevel=1) as stream:
        for p in sorted(package.iterdir()): stream.add(p,arcname=p.name)
    write(root/'export_receipt.json',dict(files=list(manifest),archive_sha256=digest(root/'package.tar.gz'),
        manifest_sha256=digest(package/'manifest.json'),fits=270,jobs=30,package_bytes=(root/'package.tar.gz').stat().st_size))
    print(dict(root=str(root),fits=270,jobs=30,accounts=784,new_hypotheses=3392),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    prepare(parser.parse_args().root)
