"""Registered breadth changes with the same 60% sum of stock targets.

This changes portfolio construction only. Source predictions, the 80% gross
ceiling, delayed execution, locked shares and all contest limits stay frozen.
"""
from quant.timefolio_heatmap_fleet_accounts import account_id, policy_grid

MEMBERS = ['seed17', 'seed29', 'seed43', 'ensemble']
WIDTHS = [(20, 300), (30, 200), (40, 150)]


def policies():
    return [dict(policy, top_n=count, weight=bp / 10000, weight_bps=bp,
                 reference_buffer=policy['buffer'],
                 buffer=count * (policy['buffer'] // 12))
            for count, bp in WIDTHS for policy in policy_grid()]


def identity(model, policy):
    return (f"width_{model}__n{policy['top_n']}__weight{policy['weight_bps']}bp"
            f"__orders{policy['max_orders']}__refresh{policy['refresh']}"
            f"__buffer{policy['buffer']}__band5bp__sector_{policy['ceiling']}")


def reference(model, policy):
    return account_id(model, dict(policy, buffer=policy['reference_buffer']))


def hypotheses(cases):
    family = []
    for case in cases:
        for member in MEMBERS:
            for policy in policies():
                trained = identity(f'{case}_trained_{member}', policy)
                untrained = identity(f'{case}_untrained_{member}', policy)
                family.extend([
                    (trained, 'cash', None),
                    (trained, 'matching_untrained_breadth', untrained),
                    (trained, 'same_breadth_nonimage', identity('online_nonimage', policy)),
                    (trained, 'original_top12', reference(f'flt_{case}_trained_{member}', policy)),
                    (untrained, 'cash', None),
                ])
    family.extend((identity('online_nonimage', policy), 'cash', None) for policy in policies())
    if len({(key, label) for key, label, _ in family}) != len(family):
        raise ValueError('Duplicate breadth hypotheses')
    return family
