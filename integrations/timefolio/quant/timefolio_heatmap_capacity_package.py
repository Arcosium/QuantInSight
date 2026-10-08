"""Register216 capacity/full-input fits before their prediction results."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import torch
from quant.timefolio_heatmap_gpu_worker import digest, verify, write
from quant.timefolio_heatmap_capacity_models import cases, LabNet

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'vault/ArcTrade/timefolio_heatmap'
ROOT = BASE / '20260929_capacity_lab_v1'
CONTROLS = ['cnn_width8','cnn_width32','summary_mlp','full_linear','full_mlp128']


def prepare(root):
    source = BASE / '20260929_geometry_lab_v2/package'; verify(source)
    parent = json.loads((source / 'plan.json').read_text())
    benchmark = BASE / '20260929_paper_benchmark_v1/registration.json'
    refs = json.loads(benchmark.read_text())['references']
    if root.exists(): raise RuntimeError('Fresh capacity root required')
    package = root / 'package'; package.mkdir(parents=True)
    for name in ['images_history.npy','labels_masks.npz','reference.py']:
        shutil.copyfile(source / name, package / name)
    files = {'engine.py':'timefolio_heatmap_gpu_worker.py', 'base_models.py':'timefolio_heatmap_lab_models.py',
             'models.py':'timefolio_heatmap_capacity_models.py', 'worker.py':'timefolio_heatmap_lab_worker.py',
             'scheduler.py':'timefolio_heatmap_lab_queue.py'}
    for target, name in files.items(): shutil.copyfile(PROJECT / 'quant' / name, package / target)
    configs = cases()
    # Static interleaving of large CNNs and smaller controls; no score-based scheduling.
    order = ['cnn_width96','full_linear','cnn_width64','summary_mlp','cnn_width32','full_mlp128','cnn_width8','full_mlp256']
    assert set(order) == {c['id'] for c in configs}
    jobs = [dict(id=f'{name}_seed{s}', case=name, seed=s) for s in [17,29,43] for name in order]
    counts = {c['id']:sum(p.numel() for p in LabNet(c).parameters()) for c in configs}
    hypotheses = sum((3 + len([x for x in CONTROLS if x != c['id']])) * 4 * 16 + 4 * 8 for c in configs)
    assert hypotheses == 4032
    plan = dict(cases=configs, seeds=[17,29,43], jobs=jobs, folds=parent['folds'], boundaries=parent['boundaries'],
        fits=216, source_period=parent['source_period'], selection=parent['selection'],
        fixed_account_grid=parent['fixed_account_grid'], parameter_counts=counts,
        paper_reference=refs,
        evaluation=dict(trained_streams=32, untrained_streams=32, online_streams=1, accounts=1040,
            new_accounts=1040, legacy_accounts=0, prior_hypotheses=56944, new_hypotheses=4032, joint_hypotheses=60976,
            members=['seed17','seed29','seed43','ensemble'], comparators=['cash','own_untrained','online']+CONTROLS,
            omit_self=True, formula_also_own_research20=True,
            bootstrap=dict(blocks=[5,10], draws=4000, seed=57, alpha=.025),
            news_family=dict(hypotheses=147, alpha=.025, separate=True)),
        execution='Every account and control recomputed under locked_repair=True.',
        rationale='The source paper uses about750k CNN parameters and a flattened-matrix MLP control. This study brackets that CNN capacity and adds full2080-code linear/MLP controls to the existing96-summary MLP. It is not an exact paper replication or an exactly parameter-matched architecture test.',
        information='Same46596 history images, sample axes, H10 labels, monthly purging, pairwise loss, lr0.0007, max6epochs and optimizer as the existing geometry controls.',
        interpretation='Geometry and optimizer performance was already reviewed before this registration; pending feature outcomes were not used to choose these models. All eight models and all seeds are retained.',
        gate=parent['gate']+' Numeric crypto-paper proximity is recorded separately and cannot override these gates.',
        limitations=parent['limitations'], export=parent['export'], billing=parent['billing'])
    write(package / 'plan.json', plan)
    evidence = [Path(__file__), source / 'manifest.json', benchmark,
                PROJECT / 'tests/test_timefolio_capacity_models.py'] + [PROJECT / 'quant' / n for n in files.values()]
    write(root / 'provenance.json', dict(hashes={str(p.resolve()):digest(p) for p in evidence},
        prior_hypotheses=56944, prior_geometry_and_optimizer_performance_read=True,
        feature_account_performance_read=False, own_prediction_results_read=False))
    manifest = {p.name:digest(p) for p in sorted(package.iterdir())}; write(package / 'manifest.json', manifest)
    with tarfile.open(root / 'package.tar.gz', 'w:gz', compresslevel=1) as stream:
        for p in sorted(package.iterdir()): stream.add(p, arcname=p.name)
    write(root / 'export_receipt.json', dict(files=list(manifest), archive_sha256=digest(root / 'package.tar.gz'),
        manifest_sha256=digest(package / 'manifest.json'), fits=216, jobs=24,
        package_bytes=(root / 'package.tar.gz').stat().st_size))
    print(dict(root=str(root), fits=216, jobs=24, parameter_counts=counts, new_hypotheses=4032), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root)
