"""Full-input controls and larger CNNs on the unchanged32x65 history image."""
from torch import nn
if __package__:
    from quant.timefolio_heatmap_lab_models import LabNet as BaseNet
else:
    from base_models import LabNet as BaseNet


def cases():
    base = dict(kind='cnn', encoding='history', width=8, kernel=[3,3], dropout=.1,
                pool='both', layout='standard', lr=.0007, epochs=6)
    configs = [dict(base, id=f'cnn_width{w}', width=w) for w in [8,32,64,96]]
    configs += [dict(base, id='summary_mlp', kind='mlp'),
                dict(base, id='full_linear', kind='full_linear'),
                dict(base, id='full_mlp128', kind='full_mlp', hidden=[128,64]),
                dict(base, id='full_mlp256', kind='full_mlp', hidden=[256,128])]
    assert len(configs) == 8
    return configs


class LabNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg['layout'] != 'standard': raise ValueError('Capacity study preserves input layout')
        self.kind = cfg['kind']
        if self.kind in ['cnn', 'mlp']:
            self.base = BaseNet(cfg)
        elif self.kind == 'full_linear':
            self.full = nn.Sequential(nn.Flatten(), nn.Linear(32*65, 1))
        elif self.kind == 'full_mlp':
            a,b = cfg['hidden']
            self.full = nn.Sequential(nn.Flatten(), nn.Linear(32*65, a), nn.LeakyReLU(.1),
                                      nn.Dropout(cfg['dropout']), nn.Linear(a,b), nn.LeakyReLU(.1),
                                      nn.Linear(b,1))
        else: raise ValueError('Unknown capacity model')

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1,32,65): raise ValueError('Fixed32x65 one-channel images required')
        if self.kind in ['cnn', 'mlp']: return self.base(x)
        return self.full(x).squeeze(1)
