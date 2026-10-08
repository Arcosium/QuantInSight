import pandas as pd
import pytest
from quant.timefolio_heatmap_transfer import MEMBERS,VARIANTS,identity
from quant.timefolio_heatmap_transfer_inference import candidates
from quant.timefolio_heatmap_fleet_accounts import policy_grid


def fixture():
    rows=[];tests=[]
    for v in VARIANTS:
        for p in policy_grid():
            for m in MEMBERS:
                key=identity('flt_case_trained_'+m,v,p)
                rows.append(dict(id=key,**{'return':.1},mdd=-.05,positive_blocks=3,four_week_turnover_stop=False))
                if m=='ensemble':
                    for label in ['cash','matching_untrained_transfer','same_transfer_nonimage','unchanged_original']:
                        for b in [5,10]:tests.append(dict(id=key,origin='transfer_v1',comparator=label,block=b,adjusted_p=.01,simultaneous_lower95=.001))
    return pd.DataFrame(rows),pd.DataFrame(tests)


def test_all_seeds_and_original_comparator_required():
    frame,stats=fixture();selected,basic=candidates(['case'],frame,stats)
    assert len(selected)==len(basic)==64
    frame.loc[0,'return']=-.1
    assert len(candidates(['case'],frame,stats)[0])==63
    with pytest.raises(AssertionError):candidates(['case'],frame,stats[stats.comparator!='unchanged_original'])
