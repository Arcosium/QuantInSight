import hashlib
from pathlib import Path

import pandas as pd
import pytest

from quant.timefolio_heatmap_price_cash_accounts import identity, MEMBERS, CONTROLS
from quant.timefolio_heatmap_price_cash_inference import candidates, ORIGIN, PARENT_SHA256
from quant.timefolio_heatmap_fleet_accounts import policy_grid


def test_price_only_inference_keeps_all_comparisons_and_seed_gates():
    parent = Path(__file__).parents[1]/'quant/timefolio_heatmap_cash_inference.py'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == PARENT_SHA256
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
    frame, stats = pd.DataFrame(rows), pd.DataFrame(tests)
    selected, basic = candidates(['case'], frame, stats)
    assert len(selected) == len(basic) == 48
    key = selected[0]
    frame.loc[frame.id == key.replace('_ensemble__', '_seed17__'), 'four_week_turnover_stop'] = True
    assert key not in candidates(['case'], frame, stats)[0]
    with pytest.raises(AssertionError):
        candidates(['case'], frame, stats[stats.comparator != 'same_exposure_nonimage'])
