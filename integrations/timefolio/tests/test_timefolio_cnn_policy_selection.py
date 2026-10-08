import numpy as np
import pytest

from quant.timefolio_cnn_account_selection import AccountValidator
from quant.timefolio_cnn_policy_selection import PolicyAccountValidator
from test_timefolio_cnn_account_selection import fixture


def expanded_fixture():
    panel,index,arrays,fold=fixture()
    n,days=24,len(index['dates'])
    panel={name:np.tile(value,2) if value.ndim==1 else np.tile(value,(2,1))
           for name,value in panel.items()}
    panel['sector']=np.arange(n)
    index['codes']=[f'{i:06}' for i in range(n)]
    arrays.update(signal_index=np.repeat(np.arange(days),n),
                  security_key=np.tile(np.arange(n),days),eligible=np.ones(n*days,bool),
                  returns=np.zeros(n*days))
    return panel,index,arrays,fold


def test_fixed_buffer5_ledger_matches_existing_validator_exactly():
    panel,index,arrays,fold=expanded_fixture()
    base=AccountValidator(panel,index,arrays,fold,rank_buffer=5)
    scores=np.random.default_rng(17).normal(size=len(base.sample_ids))
    nested=PolicyAccountValidator(base).evaluate(scores)
    original=base.evaluate(scores)
    candidate=nested['candidate_policies'][0]
    assert {k:v for k,v in candidate.items() if k!='rank_buffer'}==original
    assert nested['selected_policy']['rank_buffer'] in (5,20,50)
    assert len(nested['candidate_policies'])==3


def test_lower_cost_retention_cannot_win_by_breaking_mandatory_turnover():
    panel,index,arrays,fold=expanded_fixture()
    base=AccountValidator(panel,index,arrays,fold,rank_buffer=5)
    # Alternate two disjoint leaders each week at flat prices. Holding all
    # original positions saves fees but does not meet mandatory weekly turnover.
    daily=arrays['signal_index'][base.sample_ids]
    keys=arrays['security_key'][base.sample_ids]
    scores=np.where((daily//5)%2==0,keys,-keys).astype(float)
    result=PolicyAccountValidator(base).evaluate(scores)
    fast,medium,slow=result['candidate_policies']
    assert fast['turnover_screen_passed'] is True
    assert medium['turnover_screen_passed'] is False and slow['turnover_screen_passed'] is False
    assert slow['fees_krw']<fast['fees_krw'] and slow['net_return']>fast['net_return']
    assert result['selected_policy']['rank_buffer']==5


def test_policy_choice_cannot_see_outer_prices_or_future_return_availability():
    panel,index,arrays,fold=expanded_fixture()
    first=PolicyAccountValidator(AccountValidator(panel,index,arrays,fold,rank_buffer=5))
    scores=np.random.default_rng(29).normal(size=len(first.sample_ids))
    original=first.evaluate(scores)
    arrays['returns'][:]=np.nan
    for name in ['o','close','exec_price','exec_high','exec_low']:
        panel[name][:,fold['test_start']:]*=1000
    second=PolicyAccountValidator(AccountValidator(panel,index,arrays,fold,rank_buffer=5))
    assert second.evaluate(scores)==original
    assert max(p['last_account_mark_index'] for p in original['candidate_policies'])<fold['test_start']


def test_exact_profit_ties_choose_the_predeclared_smaller_buffer():
    panel,index,arrays,fold=fixture()
    base=AccountValidator(panel,index,arrays,fold,rank_buffer=5)
    result=PolicyAccountValidator(base).evaluate(np.ones(len(base.sample_ids)))
    assert len({p['net_return'] for p in result['candidate_policies']})==1
    assert result['selected_policy']['rank_buffer']==5
    assert base.rank_buffer==5


def test_unregistered_policies_and_incomplete_forecasts_fail_closed():
    panel,index,arrays,fold=fixture()
    base=AccountValidator(panel,index,arrays,fold,rank_buffer=5)
    for policies in [(50,20,5),(5,50),(5,20,20),(5.,20,50)]:
        with pytest.raises(ValueError):
            PolicyAccountValidator(base,rank_buffers=policies)
    validator=PolicyAccountValidator(base)
    with pytest.raises(ValueError):
        validator.evaluate(np.ones(len(base.sample_ids)-1))
    bad=np.ones(len(base.sample_ids));bad[-1]=np.nan
    with pytest.raises(ValueError):
        validator.evaluate(bad)
