"""Freeze756 matched fits with causal horizon/window-specific partitions."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from quant.timefolio_heatmap_gpu_worker import digest,verify,write
from quant.timefolio_heatmap_optimizer_lab import cases
from quant.timefolio_heatmap_absolute_training import inputs
from quant.timefolio_heatmap_walkforward import continuous_targets,fold_masks
from quant.timefolio_heatmap_stock_relation_training import check_hashes,DATA_ROOT

PROJECT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'vault/ArcTrade/timefolio_heatmap'
ROOT=BASE/'20260929_optimizer_lab_v1'


def prepare(root):
    source=BASE/'20260929_geometry_lab_v2/package';verify(source)
    check_hashes(json.loads((DATA_ROOT/'data_hashes.json').read_text()))
    parent=json.loads((source/'plan.json').read_text());p,ix,ci,di,baseline,allowed=inputs()
    assert ix['dates'][0]=='20250908' and ix['dates'][-1]=='20260923' and len(ix['dates'])==255
    if root.exists():raise RuntimeError('Fresh optimizer root required')
    package=root/'package';package.mkdir(parents=True)
    for name in ['images_history.npy','reference.py']:shutil.copyfile(source/name,package/name)
    files={'engine.py':'timefolio_heatmap_gpu_worker.py','models.py':'timefolio_heatmap_lab_models.py',
        'worker.py':'timefolio_heatmap_optimizer_worker.py','training.py':'timefolio_heatmap_optimizer_lab.py',
        'scheduler.py':'timefolio_heatmap_lab_queue.py'}
    for target,name in files.items():shutil.copyfile(PROJECT/'quant'/name,package/target)
    boundaries={};partitions=[];configs=cases();dates=np.asarray(ix['dates']);labels={}
    old=np.load(source/'labels_masks.npz')
    for cfg in configs:
        h,w,v=cfg['horizon'],cfg['train_window'],cfg['validation_sessions']
        name=f'labels_h{h}_w{w}_v{v}.npz'
        if h not in labels:labels[h]=continuous_targets(p,ci,di,h,'absolute')
        y=labels[h];data=dict(y=y,di=di);boundaries[cfg['id']]={}
        if h==10:assert np.array_equal(y,baseline,equal_nan=True)
        for fold in parent['folds']:
            month=fold['month'];tr,va,rf,pr,_=fold_masks(di,y,allowed,h,fold['start'],fold['end'],w,v)
            assert tr.sum()>=500 and va.sum()>=200 and rf.sum()>=1000
            assert max(di[tr]+h)<min(di[va]) and max(di[rf]+h)<fold['start']
            assert np.array_equal(pr,old[month+'_pr'])
            masks=dict(tr=tr,va=va,rf=rf,pr=pr)
            if (h,w,v)==(10,0,20):
                for k,a in masks.items():assert np.array_equal(a,old[month+'_'+k])
            data.update({month+'_'+k:a for k,a in masks.items()})
            boundaries[cfg['id']][month]=dict(last_inner_train_label=str(max(dates[di[tr]+h])),
                first_inner_validation_signal=str(min(dates[di[va]])),last_refit_label=str(max(dates[di[rf]+h])),
                first_execution=ix['dates'][fold['start']],last_execution=ix['dates'][fold['end']-1])
            partitions.append(dict(case=cfg['id'],month=month,train_rows=int(tr.sum()),validation_rows=int(va.sum()),
                refit_rows=int(rf.sum()),train_dates=len(np.unique(di[tr])),validation_dates=len(np.unique(di[va]))))
        if not (package/name).exists():np.savez(package/name,**data)
    jobs=[dict(id=f"{c['id']}_seed{s}",case=c['id'],seed=s) for s in [17,29,43] for c in configs]
    assert len(jobs)==84
    plan=dict(cases=configs,seeds=[17,29,43],jobs=jobs,folds=parent['folds'],boundaries=boundaries,fits=756,
        source_period=parent['source_period'],fixed_account_grid=parent['fixed_account_grid'],
        selection='Past purged dailyIC selects epoch; restart/refit. H20 uses30session inner boundary to retain10 observed validation dates. Other horizons retain20session boundary. Rolling60/120 changes only training/refit lower boundary.',
        masks='H10 expanding baseline masks exactly equal prior GPU study; every horizon/window retains identical prediction coverage. Minimum500 inner training rows and200 validation rows; short H20 first fold is disclosed.',
        evaluation=dict(trained_streams=112,untrained_streams=8,online_streams=1,accounts=1936,new_accounts=1936,legacy_accounts=0,
            prior_hypotheses=24224,new_hypotheses=9728,joint_hypotheses=33952,
            members=['seed17','seed29','seed43','ensemble'],
            comparators=['cash','own_architecture_untrained','online','matched_other_architecture','same_architecture_h10_rank_base'],
            omit_self=True,formula_also_own_research20=True,bootstrap=dict(blocks=[5,10],draws=4000,seed=57,alpha=.025),
            news_family=dict(hypotheses=147,alpha=.025,separate=True)),
        execution='Same registered grid with locked_repair=True from separate locked_replay version. Every trained/control account including online is recomputed. All known risk-repair limitations, buy audits and closing reviews remain required.',
        untrained='All84 jobs save pretraining weights/predictions; require exact equality across optimizer/horizon variants within architecture/seed. Canonical2architectures x4members produce8 unique controls.',
        gate='Only fixed3seed ensemble may qualify; everyseed andensemble positive return,MDD>-20%,>=2positivequarters,<4failedturnoverweeks. Every comparator/bothblocks adjustedp<.025 and simultaneouslower95>0. Full-cohort locked-repair comparison,stress and independentconfirmation remain required.',
        interpretation='Fourteen onefactor training conditions, each matchedCNN andMLP. Input history and geometry fixed. No result-selected training cases, seeds or signs.',
        limitations=parent['limitations'],export=parent['export'],billing=parent['billing'])
    write(package/'plan.json',plan);write(root/'partition_audit.json',partitions)
    evidence=[Path(__file__),source/'manifest.json',DATA_ROOT/'data_hashes.json',PROJECT/'tests/test_timefolio_optimizer_lab.py',
        PROJECT/'quant/timefolio_heatmap_locked_replay.py',PROJECT/'quant/timefolio_heatmap_locked_targets.py',PROJECT/'tests/test_timefolio_locked_targets.py']
    evidence += [PROJECT/'quant'/name for name in files.values()]
    write(root/'provenance.json',dict(hashes={str(p.resolve()):digest(p) for p in evidence},performance_read_before_registration=False))
    manifest={p.name:digest(p) for p in sorted(package.iterdir())};write(package/'manifest.json',manifest)
    with tarfile.open(root/'package.tar.gz','w:gz',compresslevel=1) as stream:
        for p in sorted(package.iterdir()):stream.add(p,arcname=p.name)
    write(root/'export_receipt.json',dict(files=list(manifest),archive_sha256=digest(root/'package.tar.gz'),
        manifest_sha256=digest(package/'manifest.json'),fits=756,jobs=84,package_bytes=(root/'package.tar.gz').stat().st_size))
    print(json.dumps(dict(root=str(root),fits=756,jobs=84,minimum_train_rows=min(r['train_rows'] for r in partitions),
        minimum_validation_rows=min(r['validation_rows'] for r in partitions))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);prepare(p.parse_args().root)
