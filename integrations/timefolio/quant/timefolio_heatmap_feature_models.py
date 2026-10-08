"""Matched one-channel ablations and two-channel availability controls."""
import torch
from torch import nn
if __package__:
    from quant.timefolio_heatmap_lab_models import LabNet as BaseNet
else:
    from base_models import LabNet as BaseNet


def cases():
    variants=['full','price_only','flow_only','daily_only','no_price','no_flow','no_momentum','no_context',
        'no_constraints','no_ranks','last_day','row_center','mask_constant','mask_available']
    base=dict(width=8,kernel=[3,3],dropout=.1,pool='both',layout='standard',lr=.0007,epochs=6)
    return [dict(base,id=kind+'_'+variant,kind=kind,encoding=variant,channels=2 if variant.startswith('mask_') else 1)
        for kind in ['cnn','mlp'] for variant in variants]


class LabNet(BaseNet):
    def __init__(self,cfg):
        super().__init__(cfg);self.channels=cfg['channels']
        if self.channels not in [1,2] or cfg['kind'] not in ['cnn','mlp'] or cfg['layout']!='standard':
            raise ValueError('Registered feature architectures required')
        if self.channels==2:
            if self.kind=='cnn':self.full.net[0]=nn.Conv2d(2,8,3,padding=1)
            else:self.summary[0]=nn.Linear(192,32)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(self.channels,32,65):raise ValueError('Input channel/geometry mismatch')
        if self.channels==1:return super().forward(x)
        if self.kind=='cnn':return self.full.net(x).squeeze(1)
        a=x.reshape(len(x),64,65);summary=torch.cat([a[:,:,-1],a.mean(-1),a.std(-1,correction=0)],1)
        return self.summary(summary).squeeze(1)
