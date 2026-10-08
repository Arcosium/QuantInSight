"""Local fourth worker: fixed untrained controls and exported-input checks."""
import json
from pathlib import Path
import numpy as np
import torch
from quant.timefolio_heatmap_gpu_worker import reference, verify, digest, write, predict
from quant.timefolio_heatmap_gpu_package import ROOT

def run():
    package=ROOT/'package';verify(package);ref=reference(package)
    folder=ROOT/'local_controls';folder.mkdir();torch.set_num_threads(4)
    plan=json.loads((package/'plan.json').read_text());data=np.load(package/'labels_masks.npz')
    di,y=data['di'],data['y']
    for fold in plan['folds']:
        masks={k:data[fold['month']+'_'+k] for k in ['tr','va','rf','pr']}
        assert not (masks['tr'] & masks['va']).any()
        assert max(di[masks['tr']]+10)<min(di[masks['va']])
        assert max(di[masks['rf']]+10)<fold['start']
        assert np.isfinite(y[masks['tr']|masks['va']|masks['rf']]).all()
    index=np.arange(len(di));paths=[]
    for encoding in plan['encodings']:
        x=torch.from_numpy(np.array(np.load(package/f'images_{encoding}.npy',mmap_mode='r')[:,None],copy=True))
        for seed in plan['seeds']:
            torch.manual_seed(seed);model=ref.AlternativeNet('cnn')
            name=f'cnn_dailygpu_{encoding}_untrained_seed{seed}'
            pred=predict(model,x,index);assert np.isfinite(pred).all()
            np.save(folder/(name+'.npy'),pred);torch.save(model.state_dict(),folder/(name+'.pt'))
            paths.extend([folder/(name+'.npy'),folder/(name+'.pt')])
            print(json.dumps(dict(control=name,rows=len(pred))),flush=True)
        del x
    write(folder/'hashes.json',{p.name:digest(p) for p in paths})
    write(folder/'complete.json',dict(models=9,rows=len(di),purged_folds=9,
        package_sha256=digest(package/'manifest.json'),device='cpu',torch=str(torch.__version__),
        optimizer_steps=0,performance_not_evaluated=True))

if __name__=='__main__':run()
