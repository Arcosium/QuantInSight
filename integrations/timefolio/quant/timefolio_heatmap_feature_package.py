"""Freeze756 feature ablation fits with unchanged H10 labels and predictions."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from quant.timefolio_heatmap_gpu_worker import digest,verify,write
from quant.timefolio_heatmap_feature_ablation import VARIANTS,GROUPS,availability,transform
from quant.timefolio_heatmap_feature_models import cases
from quant.timefolio_heatmap_stock_relation_training import check_hashes,DATA_ROOT
from quant.timefolio_heatmap_study import context

PROJECT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'vault/ArcTrade/timefolio_heatmap'
ROOT=BASE/'20260929_feature_lab_v1'


def prepare(root):
    source=BASE/'20260929_geometry_lab_v2/package';verify(source)
    check_hashes(json.loads((DATA_ROOT/'data_hashes.json').read_text()))
    parent=json.loads((source/'plan.json').read_text());p,ix,ci,di=context(DATA_ROOT)
    assert len(ix['dates'])==255 and ix['dates'][-1]=='20260923'
    if root.exists():raise RuntimeError('Fresh feature root required')
    package=root/'package';package.mkdir(parents=True)
    raw=np.load(source/'images_history.npy',mmap_mode='r');assert raw.shape==(46596,32,65)
    outputs={};changed={v:0 for v in VARIANTS};missing_samples=0;missing_pixels=0
    for variant in VARIANTS:
        channels=2 if variant.startswith('mask_') else 1
        outputs[variant]=np.lib.format.open_memmap(package/f'images_{variant}.npy',mode='w+',dtype='uint8',shape=(len(raw),channels,32,65))
    for start in range(0,len(raw),256):
        end=min(start+256,len(raw));image=np.array(raw[start:end]);mask=availability(p,ci[start:end],di[start:end])
        missing_samples+=int((mask==0).any(axis=(1,2)).sum());missing_pixels+=int((mask==0).sum())
        for variant,out in outputs.items():
            values=transform(image,variant,mask);out[start:end]=values
            changed[variant]+=int(np.any(values[:,0]!=image,axis=(1,2)).sum())
        assert np.array_equal(outputs['full'][start:end,0],image)
        assert np.array_equal(outputs['mask_available'][start:end,0],image)
        assert np.array_equal(outputs['mask_constant'][start:end,0],image)
    for out in outputs.values():out.flush()
    assert missing_samples==7476 and missing_pixels==51783*13
    write(root/'feature_audit.json',dict(samples=len(raw),variants=VARIANTS,groups=GROUPS,base_channel_changed_samples=changed,
        availability_changed_samples=missing_samples,availability_missing_pixels=missing_pixels,
        constant_channel=255,neutral_ablation_code=128,latest_signal_date=ix['dates'][-1],labels_used_for_transforms=False))
    for name in ['labels_masks.npz','reference.py']:shutil.copyfile(source/name,package/name)
    files={'engine.py':'timefolio_heatmap_gpu_worker.py','base_models.py':'timefolio_heatmap_lab_models.py',
        'models.py':'timefolio_heatmap_feature_models.py','worker.py':'timefolio_heatmap_feature_worker.py',
        'scheduler.py':'timefolio_heatmap_feature_queue.py'}
    for target,name in files.items():shutil.copyfile(PROJECT/'quant'/name,package/target)
    configs=cases();assert {c['encoding'] for c in configs}==set(VARIANTS)
    jobs=[dict(id=f"{c['id']}_seed{s}",case=c['id'],seed=s) for s in [17,29,43] for c in configs]
    plan=dict(cases=configs,seeds=[17,29,43],jobs=jobs,folds=parent['folds'],boundaries=parent['boundaries'],fits=756,
        source_period=parent['source_period'],selection=parent['selection'],fixed_account_grid=parent['fixed_account_grid'],
        representations=dict(groups=GROUPS,variants=VARIANTS,neutral_code=128,
            daily_availability='Extra channel255 observed and0 missing for original15 daily rows over5 days, repeated13slots; other17 rows constant255. Intraday missingness is not added.',
            matched_mask_control='Same two-channel network and unchanged image with extra channel fixed255. Initial states must match the availability model for each architecture/seed.',
            row_center='Subtract each row mean in stored-code units using float64, rescale and quantize; constant rows exactly128. No dataset-wide fitted normalization.'),
        evaluation=dict(trained_streams=112,untrained_streams=112,online_streams=1,accounts=3600,new_accounts=3600,legacy_accounts=0,
            prior_hypotheses=33952,new_hypotheses=9856,joint_hypotheses=43808,members=['seed17','seed29','seed43','ensemble'],
            comparators=['cash','own_untrained','online','matched_other_architecture','same_architecture_full'],
            availability_also_same_architecture_constant_mask=True,omit_self=True,formula_also_own_research20=True,
            bootstrap=dict(blocks=[5,10],draws=4000,seed=57,alpha=.025),news_family=dict(hypotheses=147,alpha=.025,separate=True)),
        execution='Registered grid with locked_repair=True for all models/controls; no existing account reused without replay.',
        interpretation='Fourteen fixed representations with matchedCNN andMLP. All models use baselineH10pairwise,lr0.0007,max6epochs,decay0.001. No choice based on pending geometry/optimizer performance.',
        gate='Fixed3seed ensemble and everyseed positive return,MDD>-20%,>=2positivequarters,<4failedturnoverweeks; all comparisons at bothblocks adjustedp<.025 and simultaneouslower95>0. Main audit,full-cohort rule comparison,stress and independentconfirmation required.',
        limitations=parent['limitations']+' Availability refers to archived daily-feature finiteness, not real-time data outage proof; masking changes available information, not causal economic attribution.',
        export=parent['export'],billing=parent['billing'])
    write(package/'plan.json',plan)
    evidence=[Path(__file__),source/'manifest.json',DATA_ROOT/'data_hashes.json',PROJECT/'quant/timefolio_heatmap_feature_ablation.py',
        PROJECT/'quant/timefolio_heatmap_daily_history.py',PROJECT/'quant/timefolio_heatmap_locked_replay.py',PROJECT/'tests/test_timefolio_feature_ablation.py']
    evidence += [PROJECT/'quant'/name for name in files.values()]
    write(root/'provenance.json',dict(hashes={str(p.resolve()):digest(p) for p in evidence},performance_read_before_registration=False))
    manifest={p.name:digest(p) for p in sorted(package.iterdir())};write(package/'manifest.json',manifest)
    with tarfile.open(root/'package.tar.gz','w:gz',compresslevel=1) as stream:
        for p in sorted(package.iterdir()):stream.add(p,arcname=p.name)
    write(root/'export_receipt.json',dict(files=list(manifest),archive_sha256=digest(root/'package.tar.gz'),
        manifest_sha256=digest(package/'manifest.json'),fits=756,jobs=84,package_bytes=(root/'package.tar.gz').stat().st_size))
    print(json.dumps(dict(root=str(root),fits=756,jobs=84,missing_samples=missing_samples,
        package_bytes=(root/'package.tar.gz').stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);prepare(p.parse_args().root)
