"""Monthly heatmap selection from strictly prior costed shadow returns.

Choices alter signals in one continuously replayed account. They never splice
NAV paths, fit to the current month, or use the origin day's return.
"""
import numpy as np
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_portfolio_selector import select_past
from quant.timefolio_heatmap_fleet_accounts import account_id, policy_grid
from quant.timefolio_heatmap_fleet_blend import MEMBERS, source_names, identity as blend_identity


def rules(components):
    # Top-three of a three-component pool duplicates its existing fixed mean.
    counts = [1] if len(components) == 3 else [1, 3]
    if len(components) < 3 or len(set(components)) != len(components):
        raise ValueError('At least three unique registered components required')
    return [dict(id=f'past{window}_top{count}', lookback=window, count=count)
            for window in [20, 60] for count in counts]


def identity(pool, rule, objective, member, policy):
    return account_id(f'select_{pool}_{rule}_{objective}_{member}', policy)


def hypotheses(pools):
    family = []
    for pool, components in pools.items():
        for rule in rules(components):
            for member in MEMBERS:
                for policy in policy_grid():
                    key = identity(pool, rule['id'], 'trained', member, policy)
                    control = identity(pool, rule['id'], 'untrained', member, policy)
                    family.extend([(key, 'cash', None), (key, 'matched_untrained_selector', control),
                        (key, 'online_nonimage', account_id('online_nonimage', policy)),
                        (key, 'fixed_mean_blend', blend_identity(pool, 'mean', 'trained', member, policy)),
                        (control, 'cash', None)])
                    family.extend((key, 'component_'+case, account_id(f'flt_{case}_trained_{member}', policy))
                                  for case in components)
    assert len(family) == len({(key, label) for key, label, _ in family})
    return family


def monthly_scores(matrices, eligible, dates, components, objective, member,
                   shadow_returns, shadow_dates, *, lookback, count,
                   start='20260101', end='20260923'):
    dates = np.asarray(dates)
    if dates.ndim != 1 or np.any(dates[1:] <= dates[:-1]):
        raise ValueError('Unique ordered market dates required')
    names = source_names(components, objective, member)
    if set(matrices) != set(names) or set(shadow_returns) != set(components):
        raise ValueError('Complete registered component scores and shadow returns required')
    # Validate common finite support before any member can be selected.
    rank_consensus(matrices, {name: 'component' for name in names}, eligible, 'mean')
    ranks = {name: rank_consensus({name: a}, {name: 'component'}, eligible, 'mean')
             for name, a in matrices.items()}
    if np.asarray(eligible).shape[1] != len(dates):
        raise ValueError('Score and market calendars differ')
    result = np.full(np.asarray(eligible).shape, np.nan, np.float32)
    choices = []; selected_names = None; month = None
    for day, execution in enumerate(dates):
        if day == 0 or not start <= execution <= end: continue
        if execution[:6] != month:
            month = execution[:6]
            selected, proof = select_past(components, shadow_returns, shadow_dates,
                                          execution, lookback, count)
            selected_names = sorted(source_names(selected, objective, member))
            choices.append(dict(month=str(month), first_execution=str(execution),
                                chosen=selected, **proof))
        result[:, day-1] = np.mean([ranks[name][:, day-1] for name in selected_names], axis=0)
    return result, choices
