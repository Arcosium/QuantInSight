"""Registered horizon, objective and optimizer comparisons on fixed images."""
import numpy as np
import torch
from torch import nn
if __package__:
    from quant.timefolio_heatmap_gpu_worker import predict
else:
    from engine import predict


def cases():
    base=dict(encoding='history',width=8,kernel=[3,3],dropout=.1,pool='both',layout='standard',
        lr=.0007,epochs=6,weight_decay=.001,objective='pairwise',target='absolute',horizon=10,
        train_window=0,validation_sessions=20)
    variants=[('h10_rank_base',{}),('h3_rank',dict(horizon=3)),('h5_rank',dict(horizon=5)),
        ('h20_rank',dict(horizon=20,validation_sessions=30)),('h10_huber',dict(objective='huber')),
        ('h10_mse',dict(objective='mse')),('h10_recent60',dict(train_window=60)),('h10_recent120',dict(train_window=120)),
        ('h10_lr0002',dict(lr=.0002)),('h10_lr002',dict(lr=.002)),('h10_epochs3',dict(epochs=3)),
        ('h10_epochs12',dict(epochs=12)),('h10_decay0',dict(weight_decay=0.)),('h10_decay01',dict(weight_decay=.01))]
    return [dict(base,**changes,id=kind+'_'+name,kind=kind) for kind in ['cnn','mlp'] for name,changes in variants]


def train(ref,x,y,train_mask,val,cfg,di,*,epochs=None):
    """Original date-balanced loop, varying only registered optimizer/loss fields."""
    seed=cfg['seed'];torch.manual_seed(seed)
    model=ref.AlternativeNet(cfg['architecture']).to(x.device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],foreach=False)
    target=torch.as_tensor(np.nan_to_num(y),dtype=torch.float32,device=x.device)
    groups=ref.date_groups(di,train_mask);va=np.flatnonzero(val)
    if not groups or max(map(len,groups))>512:raise ValueError('Invalid date groups')
    if not np.isfinite(y[train_mask|val]).all():raise ValueError('Nonfinite selected label')
    if np.any(train_mask&val):raise ValueError('Overlapping training and validation')
    if cfg['objective'] not in ['pairwise','huber','mse']:raise ValueError('Unknown objective')
    best,state,best_epoch,history=-np.inf,None,0,[]
    for epoch in range(epochs or cfg['epochs']):
        model.train();losses=[]
        for group in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ids=groups[group];opt.zero_grad(set_to_none=True)
            out=model(x[ids].float().div(127.5).sub(1))
            if cfg['objective']=='pairwise':loss=ref.pairwise_loss(out,target[ids])
            elif cfg['objective']=='huber':loss=nn.functional.smooth_l1_loss(out,target[ids],beta=1.)
            else:loss=nn.functional.mse_loss(out,target[ids])
            if not torch.isfinite(loss):raise AssertionError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        metric=0.
        if len(va):
            values=np.full(len(y),np.nan);values[va]=predict(model,x,va);metric=ref.daily_ic(y,values,val,di)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),inner_ic=metric))
        if not len(va) or metric>best:
            best,best_epoch=metric,epoch+1;state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model,dict(best_epoch=best_epoch,history=history,training_rows=int(train_mask.sum()),
        validation_rows=int(val.sum()),training_dates=len(groups))
