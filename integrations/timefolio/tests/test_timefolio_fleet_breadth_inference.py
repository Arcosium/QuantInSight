import pandas as pd
import pytest

from quant.timefolio_heatmap_fleet_breadth import identity, policies, MEMBERS
from quant.timefolio_heatmap_fleet_breadth_inference import candidates, should_resample, ORIGIN


def test_original_policy_and_all_controls_must_pass_with_stable_seeds():
    rows, tests = [], []
    for policy in policies():
        for member in MEMBERS:
            key = identity(f'case_trained_{member}', policy)
            rows.append(dict(id=key, **{'return': .1}, mdd=-.1, positive_blocks=2, four_week_turnover_stop=False))
            if member != 'ensemble': continue
            for label in ['cash', 'matching_untrained_breadth', 'same_breadth_nonimage', 'original_top12']:
                for block in [5, 10]:
                    tests.append(dict(id=key, comparator=label, origin=ORIGIN, block=block, adjusted_p=.01, simultaneous_lower95=.001))
    frame, stats = pd.DataFrame(rows), pd.DataFrame(tests)
    selected, basic = candidates(['case'], frame, stats)
    assert len(selected) == len(basic) == 48
    key = selected[0]
    changed = stats.copy()
    changed.loc[(changed.id == key) & (changed.comparator == 'original_top12'), 'adjusted_p'] = .025
    assert key not in candidates(['case'], frame, changed)[0]
    changed = frame.copy()
    changed.loc[changed.id == key.replace('_ensemble__', '_seed29__'), 'four_week_turnover_stop'] = True
    assert key not in candidates(['case'], changed, stats)[0]
    with pytest.raises(AssertionError):
        candidates(['case'], frame, stats[stats.comparator != 'same_breadth_nonimage'])


def test_any_target_prevents_deferring_full_family_resampling():
    assert not should_resample([], [])
    assert should_resample([{'id': 'record'}], [])
    assert should_resample([], [{'id': 'stop'}])
