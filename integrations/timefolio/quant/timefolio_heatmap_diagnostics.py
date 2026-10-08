"""Within-date feature permutation diagnostics for persisted monthly checkpoints.

Descriptive sensitivity only: permutations can break feature relationships and do
not establish causal importance or produce additional selected trading strategies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_walkforward import SOURCE,DEST,context,AlternativeNet,predict,continuous_targets,daily_ic


GROUPS={'price':(0,8),'volume':(8,16),'summary':(16,24),'rules':(24,32)}


def shuffle_group(x,dates,rows,seed):
    out=x.copy();rng=np.random.default_rng(seed)
    for date in np.unique(dates):
        ids=np.flatnonzero(dates==date);donors=rng.permutation(ids)
        out[ids,:,rows[0]:rows[1]]=x[donors,:,rows[0]:rows[1]]
    return out


def diagnostics(model_ids,month='202609',dest=DEST):
    dest=Path(dest);p,ix,ci,di=context(SOURCE);target=continuous_targets(p,ci,di,3,'sector')
    output=dest/'diagnostics';output.mkdir(exist_ok=True);rows=[]
    for name in model_ids:
        path=output/(name+'_'+month+'.json')
        if path.exists():rows.append(json.loads(path.read_text()));continue
        folder=dest/'models'/name;meta=json.loads((folder/(month+'.json')).read_text());cfg=meta['config']
        if cfg['architecture']=='gbm':raise ValueError('This diagnostic expects a neural checkpoint')
        checkpoint=torch.load(folder/(month+'.pt'),map_location='cpu',weights_only=True)
        model=AlternativeNet(cfg['architecture']);model.load_state_dict(checkpoint['state_dict'])
        predictions=np.load(folder/(month+'.pred.npy'));mask=np.isfinite(predictions)
        raw=np.load(dest/f"images_{cfg['encoding']}.npy",mmap_mode='r')
        x=np.array(raw[mask,None],copy=True);days=di[mask];y=target[mask]
        ids=np.arange(len(x));baseline=predict(model,torch.from_numpy(x),ids,cfg['target']=='binary')
        np.testing.assert_allclose(baseline,predictions[mask],atol=1e-6,rtol=1e-6)
        base_ic=daily_ic(y,baseline,np.isfinite(y),days);result=[]
        for group,span in GROUPS.items():
            repetitions=[]
            for seed in [11,23,37]:
                changed=shuffle_group(x,days,span,seed)
                scores=predict(model,torch.from_numpy(changed),ids,cfg['target']=='binary')
                correlations=[];overlaps=[]
                for day in np.unique(days):
                    use=np.flatnonzero(days==day)
                    correlations.append(float(spearmanr(baseline[use],scores[use]).statistic))
                    top=min(20,len(use));a=set(use[np.argsort(-baseline[use])[:top]]);b=set(use[np.argsort(-scores[use])[:top]])
                    overlaps.append(len(a&b)/top)
                repetitions.append({'seed':seed,'daily_rank_correlation':float(np.mean(correlations)),
                                    'top20_overlap':float(np.mean(overlaps)),
                                    'common_sector_ic':daily_ic(y,scores,np.isfinite(y),days)})
            result.append({'group':group,'baseline_common_sector_ic':base_ic,
                           **{key:float(np.mean([r[key] for r in repetitions])) for key in
                              ['daily_rank_correlation','top20_overlap','common_sector_ic']},'repetitions':repetitions})
        report={'model':name,'month':month,'rows':len(x),'days':len(np.unique(days)),
                'baseline_common_sector_ic':base_ic,'groups':result,
                'interpretation':'within-date stock permutation; descriptive and potentially out of distribution; observed development data; no model selection based on this diagnostic'}
        atomic_json(path,report);rows.append(report)
        print(json.dumps({'model':name,'baseline_ic':base_ic,'groups':[{k:v for k,v in r.items() if k!='repetitions'} for r in result]}),flush=True)
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--month',default='202609')
    args=ap.parse_args();torch.set_num_threads(1);diagnostics(args.models,args.month)
