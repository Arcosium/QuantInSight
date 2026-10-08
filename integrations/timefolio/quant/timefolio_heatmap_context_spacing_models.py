"""Matched two-channel networks; reuse existing stride1 fits as controls."""
if __package__:
    from quant.timefolio_heatmap_feature_models import LabNet
else:
    from feature_models import LabNet


def evaluation_cases():
    base = dict(width=8, kernel=[3, 3], dropout=.1, pool='both', layout='standard', lr=.0007, epochs=6, channels=2)
    return [dict(base, id=f'{kind}_s{stride}_{mask}', kind=kind, encoding=f's{stride}_{mask}',
                 stride=stride, availability_mask=mask)
            for kind in ['cnn', 'mlp'] for stride in [1, 2, 4] for mask in ['constant', 'available']]


def cases():
    return [c for c in evaluation_cases() if c['stride'] != 1]


def feature_control(case):
    if case['stride'] != 1:
        return None
    return case['kind'] + '_mask_' + case['availability_mask']
