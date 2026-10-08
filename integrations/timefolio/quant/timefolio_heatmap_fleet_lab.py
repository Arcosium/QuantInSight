"""Preregistered interactions, with all conditions and seeds retained."""
from itertools import product

MAX_PODS = 30
SEEDS = [17, 29, 43]
MEMBERS = ['seed17', 'seed29', 'seed43', 'ensemble']


def geometries():
    return [dict(id=f'cnn_w{w}_k{k[0]}{k[1]}_{layout}', kind='cnn', width=w,
                 kernel=k, layout=layout)
            for w, k, layout in product([8, 32], [[3, 3], [1, 5]],
                                         ['standard', 'rowshuffle', 'transpose'])] + [
        dict(id='mlp', kind='mlp', width=8, kernel=[3, 3], layout='standard')]


def cases():
    base = dict(encoding='history', dropout=.1, pool='both', lr=.0007,
                weight_decay=.001, target='absolute', train_window=0, validation_sessions=20,
                epoch_rule='fixed')
    result = []
    for geometry, horizon, objective, epochs in product(geometries(), [3, 5, 10],
                                                       ['pairwise', 'mse'], [3, 12]):
        g = dict(geometry); name = g.pop('id')
        result.append(dict(base, **g, geometry=name, horizon=horizon, objective=objective,
            epochs=epochs, id=f'{name}_h{horizon}_{objective}_e{epochs}'))
    assert len(result) == 156 and len({c['id'] for c in result}) == 156
    return result


def paired_case(case, geometry):
    return f"{geometry}_h{case['horizon']}_{case['objective']}_e{case['epochs']}"


def comparison_cases(case):
    other = 'mlp' if case['kind'] == 'cnn' else 'cnn_w8_k33_standard'
    result = [('matched_other_architecture', paired_case(case, other))]
    if case['kind'] == 'cnn' and case['geometry'] != 'cnn_w8_k33_standard':
        result.append(('matched_base_geometry', paired_case(case, 'cnn_w8_k33_standard')))
    return result


def assignments(count=10):
    if not 1 <= count <= MAX_PODS:
        raise ValueError(f'At most {MAX_PODS} Pods are authorized')
    groups = [[] for _ in range(count)]; loads = [0.] * count
    def cost(c):
        # Cost affects scheduling only; no outcomes enter assignment.
        return c['epochs'] * ((c['width'] / 8) ** 1.5 if c['kind'] == 'cnn' else .4)
    for c in sorted(cases(), key=lambda c: (-cost(c), c['id'])):
        i = min(range(count), key=lambda j: (loads[j], j))
        groups[i].append(c['id']); loads[i] += cost(c)
    return groups
