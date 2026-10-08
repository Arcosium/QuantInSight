import numpy as np
import pytest

from test_timefolio_heatmap import synthetic_panel
from quant.timefolio_heatmap_action_amendment import release_audit, amend_actions


@pytest.mark.parametrize('sale_day,invalid', [(3, True), (4, False)])
def test_announced_entitlements_cannot_sell_before_listing(sale_day, invalid):
    p,ix=synthetic_panel();code=ix['codes'][0];p['split'][0,2]=2.
    result={'daily':[{'date':d} for d in ix['dates'][1:5]],
            'trades':[{'date':ix['dates'][1],'code':code,'side':'buy','qty':10},
                      {'date':ix['dates'][sale_day],'code':code,'side':'sell','qty':11}]}
    release={(code,ix['dates'][2]):ix['dates'][4]}
    assert bool(release_audit(p,ix,result,release)) is invalid


def test_action_amendment_preserves_frozen_inputs_and_removes_false_share_reduction():
    dates=['20260504','20260604','20260630'];codes=['321370','068270']
    p={'split':np.array([[.970482,1.,1.],[1.,1.049135,1.]],dtype=np.float32)}
    original=p['split'].copy()
    amended,release=amend_actions(p,{'codes':codes,'dates':dates},[
        {'code':'321370','ex_date':'20260504','quantity_ratio':1.,'listing_date':'20260504'}])
    np.testing.assert_array_equal(p['split'],original)
    assert amended['split'][0,0]==1.
    assert amended['split'][1,1]==1.05
    assert release[('068270','20260604')]=='20260630'
