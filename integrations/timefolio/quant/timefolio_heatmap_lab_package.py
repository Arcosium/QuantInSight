"""Freeze540 additional fits and their comparisons before any account results."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from quant.timefolio_heatmap_gpu_worker import digest,verify,write
from quant.timefolio_heatmap_lab_models import cases

PROJECT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'vault/ArcTrade/timefolio_heatmap'
ROOT=BASE/'20260929_geometry_lab_v1'
CONTROLS=['history_mlp','history_linear','history_base','snapshot_base','reverse_base']


def prepare(root):
    source=BASE/'20260929_daily_history_gpu_v2/package';verify(source)
    prior=json.loads((source/'plan.json').read_text())
    if root.exists():raise RuntimeError('Fresh lab root required')
    package=root/'package';package.mkdir(parents=True)
    for name in ['images_snapshot.npy','images_history.npy','images_reverse.npy','labels_masks.npz','reference.py']:
        shutil.copyfile(source/name,package/name)
    files={'engine.py':'timefolio_heatmap_gpu_worker.py','models.py':'timefolio_heatmap_lab_models.py',
        'worker.py':'timefolio_heatmap_lab_worker.py','queue.py':'timefolio_heatmap_lab_queue.py'}
    for target,name in files.items():shutil.copyfile(PROJECT/'quant'/name,package/target)
    configs=cases();jobs=[dict(id=f"{c['id']}_seed{s}",case=c['id'],seed=s) for s in [17,29,43] for c in configs]
    new_hypotheses=sum((3+len([r for r in CONTROLS if r!=c['id']]))*4*16+4*8 for c in configs)
    assert new_hypotheses==10560
    plan=dict(cases=configs,seeds=[17,29,43],folds=prior['folds'],boundaries=prior['boundaries'],jobs=jobs,
        fits=540,source_period=prior['source_period'],selection=prior['selection'],fixed_account_grid=prior['fixed_account_grid'],
        evaluation=dict(trained_streams=80,untrained_streams=80,online_streams=1,accounts=2576,new_accounts=2560,
            legacy_accounts=16,prior_hypotheses=13664,new_hypotheses=new_hypotheses,joint_hypotheses=24224,
            members=['seed17','seed29','seed43','ensemble'],comparators=['cash','own_untrained','online']+CONTROLS,
            omit_self=True,formula_also_own_research20=True,bootstrap=dict(blocks=[5,10],draws=4000,seed=57,alpha=.025),
            news_family=dict(hypotheses=147,alpha=.025,separate=True)),
        gate='Only fixed3seed ensemble may qualify; ensemble and everyseed positive return,MDD>-20%,>=2positivequarters,<4failedturnoverweeks. Every nonself comparator at bothblocks needs adjustedp<.025 and simultaneouslower95>0. Closing audit,stress and freshconfirmation still required.',
        interpretation='Onefactor geometry/capacity tests plus alternative architectures. All20 configurations retained. Same Pod and package for everyseed; controls rerun. No performance-based scheduling or direction selection.',
        limitations=prior['limitations'],export=prior['export'],
        billing=dict(max_pods=1,max_parallel_jobs=2,max_hourly_with_storage_guard=.50,renewable_lease_hours=6,
            batch_completion_terminates=False,auto_recharge=False))
    write(package/'plan.json',plan)
    evidence=[source/'manifest.json',Path(__file__),PROJECT/'tests/test_timefolio_lab_models.py']
    evidence += [PROJECT/'quant'/name for name in files.values()]
    write(root/'provenance.json',dict(hashes={str(p):digest(p) for p in evidence},prior_family=13664,
        account_performance_read_before_registration=False))
    manifest={p.name:digest(p) for p in sorted(package.iterdir())};write(package/'manifest.json',manifest)
    with tarfile.open(root/'package.tar.gz','w:gz',compresslevel=1) as stream:
        for p in sorted(package.iterdir()):stream.add(p,arcname=p.name)
    write(root/'export_receipt.json',dict(files=list(manifest),archive_sha256=digest(root/'package.tar.gz'),
        manifest_sha256=digest(package/'manifest.json'),fits=540,jobs=60,package_bytes=(root/'package.tar.gz').stat().st_size))
    print(json.dumps(dict(root=str(root),fits=540,jobs=60,hypotheses=24224)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);prepare(p.parse_args().root)
