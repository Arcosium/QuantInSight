"""Fixed chart-image expansion; registration never depends on experiment scores."""
from itertools import product
from quant.timefolio_heatmap_fleet_lab import MAX_PODS, SEEDS, MEMBERS

VIEWS = ['ohlc20', 'candle20', 'line20', 'candle60', 'multi20_60', 'intraday5']
SHAPE = (64, 192)


def cases():
    return [dict(id=f'{view}_w{width}_h{horizon}_{objective}_e{epochs}',
                 view=view, width=width, horizon=horizon, objective=objective,
                 epochs=epochs, kind='chart_cnn', architecture='chart_cnn',
                 encoding=view, lr=.0007, weight_decay=.001, dropout=.1,
                 target='absolute', train_window=0, validation_sessions=20,
                 epoch_rule='fixed', image_shape=list(SHAPE))
            for view, width, horizon, objective, epochs in product(
                VIEWS, [8, 32], [3, 5, 10], ['pairwise', 'mse', 'bce'], [3, 12])]


def reference_case(case):
    return f"candle20_w{case['width']}_h{case['horizon']}_{case['objective']}_e{case['epochs']}"


def scheduling_cost(case):
    # Configuration-only scheduling; observed returns never determine priority.
    return case['epochs'] * (case['width'] / 8) ** 1.5


def assignments(count=30):
    if not 1 <= count <= MAX_PODS:
        raise ValueError('Pod count outside authorized range')
    groups = [[] for _ in range(count)]; loads = [0.] * count
    for case in sorted(cases(), key=lambda c: (-scheduling_cost(c), c['id'])):
        i = min(range(count), key=lambda j: (loads[j], j))
        groups[i].append(case['id']); loads[i] += scheduling_cost(case)
    return groups
