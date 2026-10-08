"""Compare fixed refit durations with the original inner-IC epoch selection."""
import numpy as np
if __package__:
    from quant.timefolio_heatmap_gpu_worker import train
else:
    from engine import train


def cases():
    base = dict(encoding='history', width=8, kernel=[3,3], dropout=.1,
                pool='both', layout='standard', lr=.0007)
    return [dict(base, id=kind+'_'+name, kind=kind, epoch_rule=rule, epochs=count)
            for kind in ['cnn', 'mlp'] for name, rule, count in
            [('selected6', 'inner_ic', 6), ('fixed1', 'fixed', 1), ('fixed3', 'fixed', 3),
             ('fixed6', 'fixed', 6), ('fixed12', 'fixed', 12)]]


def fit_fold(ref, x, y, masks, cfg, di):
    if cfg['epoch_rule'] == 'inner_ic':
        candidate, selection = train(ref, x, y, masks['tr'], masks['va'], cfg, di)
        del candidate
        count = selection['best_epoch']
    elif cfg['epoch_rule'] == 'fixed':
        selection = None; count = cfg['epochs']
    else:
        raise ValueError('Unregistered epoch selection rule')
    model, refit = train(ref, x, y, masks['rf'], np.zeros(len(y), bool), cfg, di, epochs=count)
    assert refit['best_epoch'] == count and refit['validation_rows'] == 0
    return model, dict(fit=selection, refit_fit=refit, selection_rule=cfg['epoch_rule'],
                       refit_epochs=count, inner_validation_used=selection is not None)
