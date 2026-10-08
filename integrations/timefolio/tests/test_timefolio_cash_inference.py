import pandas as pd
import pytest

from quant.timefolio_heatmap_cash_accounts import identity, MEMBERS, CONTROLS
from quant.timefolio_heatmap_cash_inference import candidates, ORIGIN
from quant.timefolio_heatmap_fleet_accounts import policy_grid


def fixture():
    rows, tests = [], []
    for policy in policy_grid():
        for control in CONTROLS:
            for member in MEMBERS:
                key = identity('case', 'trained', member, policy, control)
                rows.append(dict(id=key, **{'return':.1}, mdd=-.1, positive_blocks=2, four_week_turnover_stop=False))
                if member != 'ensemble': continue
                for comparator in ['cash', 'unchanged_baseline', 'same_exposure_nonimage', 'matching_untrained_pipeline']:
                    for block in [5,10]:
                        tests.append(dict(id=key, comparator=comparator, origin=ORIGIN, block=block,
                            adjusted_p=.01, simultaneous_lower95=.001))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def test_every_seed_and_both_blocks_are_required():
    frame, stats = fixture()
    selected, basic = candidates(['case'], frame, stats)
    assert len(selected) == len(basic) == 112
    key = selected[0]
    stats.loc[(stats.id == key) & (stats.block == 10), 'adjusted_p'] = .025
    assert key not in candidates(['case'], frame, stats)[0]
    frame.loc[frame.id == key.replace('_ensemble__', '_seed29__'), 'return'] = 0.
    assert key not in candidates(['case'], frame, stats)[1]
    with pytest.raises(AssertionError):
        candidates(['case'], frame, stats.iloc[1:])
