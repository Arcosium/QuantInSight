import numpy as np
import pytest
from quant.timefolio_heatmap_fleet_accounts import account_id, policy_grid
from quant.timefolio_heatmap_fleet_selector import identity, hypotheses
from quant.timefolio_heatmap_fleet_selector_accounts import units, shadow_for


def test_queue_covers_each_registered_account_once_in_small_units():
    pools={str(i):list('abc') for i in range(4)};pools['broad']=list('abcdefghi')
    jobs=units(pools);assert len(jobs)==96
    accounts=[identity(j['pool'],j['rule']['id'],j['objective'],j['member'],policy)
              for j in jobs for policy in policy_grid()]
    assert len(accounts)==len(set(accounts))==1536
    assert set(accounts)=={key for key,_,_ in hypotheses(pools)}


def test_shadow_policy_and_member_are_matched_instead_of_reusing_best_seed():
    policy=policy_grid()[0]
    ids=[account_id('flt_case_trained_'+member,policy) for member in ['seed17','ensemble']]
    values=np.column_stack([np.ones(179),np.full(179,7.)])
    actual=shadow_for(['case'],'trained','ensemble',policy,{'case':(ids,values)})
    np.testing.assert_array_equal(actual['case'],np.full(179,7.))
    with pytest.raises(AssertionError):
        shadow_for(['case'],'untrained','ensemble',policy,{'case':(ids,values)})
