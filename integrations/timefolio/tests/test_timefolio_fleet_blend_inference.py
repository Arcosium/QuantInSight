import pandas as pd
import pytest

from quant.timefolio_heatmap_fleet_blend import identity, MEMBERS, METHODS
from quant.timefolio_heatmap_fleet_blend_inference import candidates, ORIGIN
from quant.timefolio_heatmap_fleet_accounts import policy_grid


def test_consensus_must_beat_every_component_and_have_stable_seeds():
    pools = {'three': ['a', 'b', 'c'], 'four': ['a', 'b', 'c', 'd']}
    rows, tests = [], []
    for pool, components in pools.items():
        for method in METHODS:
            for policy in policy_grid():
                for member in MEMBERS:
                    key = identity(pool, method, 'trained', member, policy)
                    rows.append(dict(id=key, **{'return': .1}, mdd=-.1, positive_blocks=2, four_week_turnover_stop=False))
                    if member != 'ensemble': continue
                    for comparator in ['cash', 'matched_untrained_blend', 'online_nonimage'] + ['component_'+c for c in components]:
                        for block in [5, 10]:
                            tests.append(dict(id=key, comparator=comparator, origin=ORIGIN, block=block,
                                adjusted_p=.01, simultaneous_lower95=.001))
    frame, stats = pd.DataFrame(rows), pd.DataFrame(tests)
    selected, basic = candidates(pools, frame, stats)
    assert len(selected) == len(basic) == 64
    key = selected[0]
    changed = stats.copy()
    changed.loc[(changed.id == key) & (changed.comparator == 'component_c') & (changed.block == 10), 'adjusted_p'] = .025
    assert key not in candidates(pools, frame, changed)[0]
    changed = frame.copy()
    changed.loc[changed.id == key.replace('_ensemble__', '_seed29__'), 'four_week_turnover_stop'] = True
    assert key not in candidates(pools, changed, stats)[0]
    with pytest.raises(AssertionError):
        candidates(pools, frame, stats[stats.comparator != 'component_d'])
