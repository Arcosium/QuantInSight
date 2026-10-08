"""One registered model/seed job, suitable for a persistent two-worker GPU Pod."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
if __package__:
    from quant import timefolio_heatmap_gpu_worker as engine
    from quant.timefolio_heatmap_lab_models import LabNet
    from quant.timefolio_heatmap_epoch_lab import cases,fit_fold
else:
    import engine
    from models import LabNet
    from training import cases,fit_fold


def run(root,out,case,seed):
    root,out=Path(root),Path(out);engine.verify(root);engine.configure('cuda')
    plan=json.loads((root/'plan.json').read_text())
    if plan['cases']!=cases() or seed not in [17,29,43]:raise ValueError('Unregistered lab configuration')
    template=next(c for c in cases() if c['id']==case)
    cfg=dict(template,id=f'{case}_seed{seed}',architecture=template['kind'],objective='pairwise',target='absolute',horizon=10,seed=seed)
    if out.exists():raise RuntimeError('Do not overwrite a lab job')
    out.mkdir(parents=True);model_root=out/'models'/cfg['id'];model_root.mkdir(parents=True)
    ref=engine.reference(root);ref.AlternativeNet=lambda architecture:LabNet(cfg)
    data=np.load(root/'labels_masks.npz',allow_pickle=False);y,di=data['y'],data['di']
    raw=np.load(root/f"images_{cfg['encoding']}.npy",mmap_mode='r')
    x=torch.from_numpy(np.array(raw[:,None],copy=True)).to('cuda')
    started=time.time();manifest_sha=engine.digest(root/'manifest.json')
    engine.write(out/'runtime.json',dict(config=cfg,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        numpy=np.__version__,device='cuda',package_sha256=manifest_sha,started_at=started))
    torch.manual_seed(seed);untrained=LabNet(cfg).cuda()
    values=engine.predict(untrained,x,np.arange(len(y)));assert np.isfinite(values).all()
    torch.save({k:v.cpu() for k,v in untrained.state_dict().items()},out/'untrained.pt')
    np.save(out/'untrained.pred.npy',values);del untrained
    for i,fold in enumerate(plan['folds']):
        month=fold['month'];m={k:data[month+'_'+k] for k in ['tr','va','rf','pr']};tick=time.monotonic()
        model,training=fit_fold(ref,x,y,m,cfg,di)
        forecast=np.full(len(y),np.nan,np.float32);forecast[m['pr']]=engine.predict(model,x,np.flatnonzero(m['pr']))
        if not np.array_equal(np.isfinite(forecast),m['pr']):raise AssertionError('Prediction coverage')
        path=model_root/month
        torch.save(dict(config=cfg,state_dict={k:v.cpu() for k,v in model.state_dict().items()}),path.with_suffix('.pt'));del model
        np.save(path.with_suffix('.pred.npy'),forecast)
        engine.write(path.with_suffix('.json'),dict(config=cfg,fold=fold,**training,refit_rows=int(m['rf'].sum()),
            boundaries=plan['boundaries'][month],**plan['boundaries'][month],seconds=time.monotonic()-tick))
        engine.write(out/'progress.json',dict(model=cfg['id'],completed=i+1,expected=9,month=month))
        print(json.dumps(dict(model=cfg['id'],month=month,completed=i+1,seconds=time.monotonic()-tick)),flush=True)
    engine.verify(root)
    artifacts={str(p.relative_to(out)):engine.digest(p) for p in model_root.iterdir()}
    artifacts.update({n:engine.digest(out/n) for n in ['untrained.pt','untrained.pred.npy']});assert len(artifacts)==29
    engine.write(out/'artifact_hashes.json',artifacts)
    engine.write(out/'complete.json',dict(folds=9,model=cfg['id'],package_sha256=manifest_sha,seconds=time.time()-started,
        maximum_gpu_memory_bytes=torch.cuda.max_memory_allocated(),evaluation_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',required=True);parser.add_argument('--seed',type=int,required=True)
    a=parser.parse_args();run(a.root,a.out,a.case,a.seed)
