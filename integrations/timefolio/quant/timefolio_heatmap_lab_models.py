"""Registered geometry and capacity comparisons for fixed32x65 heatmaps."""
import numpy as np
import torch
from torch import nn


def cases():
    base=dict(kind='cnn',encoding='history',width=8,kernel=[3,3],dropout=.1,pool='both',layout='standard',lr=.0007,epochs=6)
    items=[]
    def add(name,**kwargs):items.append(dict(base,id=name,**kwargs))
    add('snapshot_base',encoding='snapshot');add('history_base');add('reverse_base',encoding='reverse')
    add('history_mlp',kind='mlp');add('history_linear',kind='linear')
    add('history_tcn',kind='tcn');add('history_split',kind='split')
    for width in [4,16,32]:add('history_width'+str(width),width=width)
    for kernel in [[1,5],[5,1],[3,5]]:add('history_kernel'+''.join(map(str,kernel)),kernel=kernel)
    add('history_timepool',pool='time')
    add('history_dropout0',dropout=0.);add('history_dropout30',dropout=.3)
    for layout in ['rowreverse','rowshuffle','interleave','transpose']:add('history_'+layout,layout=layout)
    assert len(items)==20 and len({item['id'] for item in items})==20
    return items


class FixedPool(nn.Module):
    def __init__(self,output=(2,4)):super().__init__();self.output=output
    def forward(self,x):
        h,w=x.shape[-2:];oh,ow=self.output
        if h%oh or w%ow:raise ValueError('Fixed disjoint pooling requires divisible geometry')
        return nn.functional.avg_pool2d(x,(h//oh,w//ow))


class TemporalPool(nn.Module):
    def forward(self,x):
        # Same overlapping bins as AdaptiveAvgPool1d(4), with deterministic
        # elementary reductions and slice gradients on CUDA.
        n=x.shape[-1]
        return torch.stack([x[...,i*n//4:((i+1)*n+3)//4].mean(-1) for i in range(4)],-1)


class LabNet(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=dict(cfg);self.kind=cfg['kind'];self.layout=cfg['layout']
        if self.kind!='cnn' and self.layout!='standard':raise ValueError('Non-CNN geometry must remain standard')
        order=np.arange(32)
        if self.layout=='rowreverse':order=order[::-1].copy()
        elif self.layout=='rowshuffle':order=np.random.default_rng(2718).permutation(32)
        elif self.layout=='interleave':order=order.reshape(4,8).T.ravel()
        elif self.layout not in ['standard','transpose']:raise ValueError('Unknown layout')
        self.register_buffer('row_order',torch.tensor(order.copy(),dtype=torch.int64),persistent=False)
        if self.kind=='cnn':
            width=cfg['width'];kernel=tuple(cfg['kernel']);pool=2 if cfg['pool']=='both' else (1,2)
            self.full=nn.Module()
            self.full.net=nn.Sequential(nn.Conv2d(1,width,kernel,padding=(kernel[0]//2,kernel[1]//2)),nn.LeakyReLU(.1),nn.MaxPool2d(pool),
                nn.Conv2d(width,2*width,3,padding=1),nn.LeakyReLU(.1),nn.MaxPool2d(pool),
                nn.Conv2d(2*width,4*width,3,padding=1),nn.LeakyReLU(.1),FixedPool(),nn.Flatten(),
                nn.Dropout(cfg['dropout']),nn.Linear(32*width,1))
        elif self.kind=='mlp':
            self.summary=nn.Sequential(nn.Linear(96,32),nn.LeakyReLU(.1),nn.Dropout(.1),nn.Linear(32,16),nn.LeakyReLU(.1),nn.Linear(16,1))
        elif self.kind=='linear':self.summary=nn.Linear(96,1)
        elif self.kind=='tcn':
            self.temporal=nn.Sequential(nn.Conv1d(32,16,5,padding=2),nn.LeakyReLU(.1),nn.Conv1d(16,16,3,padding=2,dilation=2),
                nn.LeakyReLU(.1),TemporalPool(),nn.Flatten(),nn.Dropout(.1),nn.Linear(64,1))
        elif self.kind=='split':
            self.price=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.LeakyReLU(.1),nn.MaxPool2d(2),
                nn.Conv2d(8,16,3,padding=1),nn.LeakyReLU(.1),nn.MaxPool2d(2),FixedPool(),nn.Flatten())
            self.summary=nn.Sequential(nn.Linear(32,32),nn.LeakyReLU(.1))
            self.head=nn.Sequential(nn.Linear(160,32),nn.LeakyReLU(.1),nn.Dropout(.1),nn.Linear(32,1))
        else:raise ValueError('Unknown registered architecture')

    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(1,32,65):raise ValueError('Fixed32x65 heatmaps required')
        if self.kind=='cnn':
            if self.layout=='transpose':x=x.transpose(-1,-2)
            elif self.layout!='standard':x=x[:,:,self.row_order,:]
            return self.full.net(x).squeeze(1)
        if self.kind=='tcn':return self.temporal(x[:,0]).squeeze(1)
        if self.kind=='split':
            a=x[:,0,16:];summary=torch.cat([a.mean(-1),a.std(-1,correction=0)],1)
            return self.head(torch.cat([self.price(x[:,:,:16]),self.summary(summary)],1)).squeeze(1)
        a=x[:,0];summary=torch.cat([a[:,:,-1],a.mean(-1),a.std(-1,correction=0)],1)
        return self.summary(summary).squeeze(1)
