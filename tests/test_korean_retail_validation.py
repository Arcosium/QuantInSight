"""Time cutoffs, paired evaluation, funded replay and honest target definitions."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.investor_society import BIAS_NAMES
from research.korean_retail_validation import (
    FEATURES, RetailReplay, autocorrelation, clean_frame, fit_engine, fit_linear,
    make_panel, paired_block_interval, predict_engine, predict_linear,
    predictive_metrics, run_analysis, split_panel,
    quote_basis_check,
)


def data_fixture(n=220):
    dates = pd.bdate_range('2024-01-01', periods=n).strftime('%Y-%m-%d').tolist()
    close = 1000 * np.cumprod(1 + .002 * np.sin(np.arange(n)))
    p = pd.DataFrame({'date': dates, 'open': close*.999, 'high': close*1.02,
                      'low': close*.98, 'close': close, 'volume': 10000+np.arange(n)})
    f = pd.DataFrame({'date': dates, 'inst_net': 100+np.arange(n), 'foreign_net': -40,
                      'individual_shares': -(60+np.arange(n))+5})
    return dates, {'005930': {'price': p, 'flow': f}}


def test_duplicate_conflicts_are_quarantined_not_arbitrarily_last_wins():
    f = pd.DataFrame({'date':['2025-01-02','2025-01-02','2025-01-03','2025-01-03','bad','2026-10-02'],
                      'inst_net':[1,1,2,3,4,5], 'foreign_net':[1,1,1,1,1,1]})
    clean, q = clean_frame(f,'flow','2026-10-02')
    assert clean.date.tolist()==['2025-01-02']
    assert q['exact_duplicate_rows_removed']==1
    assert q['conflicting_dates_quarantined']==1 and q['invalid_rows']==1


def test_invalid_market_bounds_and_zero_volume_are_excluded():
    _, markets = data_fixture(5)
    p=markets['005930']['price']
    p.loc[0,'close']=-1; p.loc[1,'high']=1; p.loc[2,'volume']=0
    clean,q=clean_frame(p,'price','2026-10-02')
    assert len(clean)==2 and q['invalid_market_rows']==3


def test_next_session_labels_use_shares_and_keep_direct_individual_distinct():
    dates, markets=data_fixture()
    panel,q=make_panel(markets,dates,None,dates[0])
    row=panel.iloc[0]; j=dates.index(row.date)
    assert row.label_date==dates[j+1]
    assert row.flow_proxy==pytest.approx(-(60+j+1)/(10000+j+1))
    assert row.individual_ratio==pytest.approx((-(60+j+1)+5)/(10000+j+1))
    assert row.individual_ratio!=row.flow_proxy


def test_missing_next_session_cannot_be_bridged_to_a_later_quote():
    dates,markets=data_fixture()
    missing=dates[100]
    markets['005930']['price']=markets['005930']['price'].query('date!=@missing')
    panel,_=make_panel(markets,dates,None,dates[0])
    assert dates[99] not in set(panel.date)


def test_future_inputs_do_not_change_past_feature_values():
    dates, markets=data_fixture()
    a,_=make_panel(markets,dates,None,dates[0]); changed=copy.deepcopy(markets)
    changed['005930']['flow'].loc[101:,'inst_net']=-3000
    changed['005930']['price'].loc[101:,'volume']=20000
    b,_=make_panel(changed,dates,None,dates[0])
    x=a[a.date==dates[100]].iloc[0];y=b[b.date==dates[100]].iloc[0]
    assert x[list(FEATURES)].tolist()==y[list(FEATURES)].tolist()
    assert x.flow_proxy!=y.flow_proxy


def test_unverified_corporate_action_jump_is_quarantined():
    dates,markets=data_fixture()
    p=markets['005930']['price']
    for c in ['open','high','low','close']:p.loc[100:,c]*=2
    panel,q=make_panel(markets,dates,None,dates[0])
    assert q['005930']['unverified_price_jumps_quarantined']==1
    assert dates[99] not in set(panel.date)
    assert not set(dates[100:121]) & set(panel.date)


def test_net_quantity_exceeding_total_market_volume_is_not_a_target():
    dates,markets=data_fixture()
    markets['005930']['flow'].loc[100,'inst_net']=1000000
    panel,q=make_panel(markets,dates,None,dates[0])
    assert q['005930']['flow_rows_exceeding_total_volume']==1
    assert dates[99] not in set(panel.date)


def test_shared_label_date_split_purges_boundary_targets():
    dates,markets=data_fixture()
    panel,_=make_panel(markets,dates,None,dates[0])
    other=panel.copy();other['code']='000660'
    split,q=split_panel(pd.concat([panel,other]))
    assert split.groupby('label_date').split.nunique().max()==1
    assert split[split.split=='train'].label_date.max()<split[split.split=='validation'].label_date.min()
    assert split[split.split=='validation'].label_date.max()<split[split.split=='test'].label_date.min()


@pytest.mark.parametrize('disabled',[(),BIAS_NAMES])
def test_funded_external_replay_preserves_cash_and_shares(disabled):
    s=RetailReplay.create(60,7,disabled)
    initial_external=s.external.shares[:]
    resident_initial=[sum(a.shares[j] for a in s.residents) for j in range(3)]
    net=np.zeros(3,dtype=int)
    for i in range(80):
        result=s.emit([round(p*(1+.1*np.sin(i/8))) for p in [1200,1000,900]])
        net+=result['paper_retail_net_shares']
        assert result['cash_error']==0 and result['share_error']==[0,0,0]
        assert len(s.residents)==60
        assert all(-1<=p<=1 for p in result['pressure'])
    assert net.tolist()==[sum(a.shares[j] for a in s.residents)-resident_initial[j] for j in range(3)]
    assert net.tolist()==[initial_external[j]-s.external.shares[j] for j in range(3)]
    assert any(net)
    s.validate_replay()


def test_unknown_fundamentals_are_masked_instead_of_fictional_company_accounts():
    s=RetailReplay.create(10,7)
    obs=s.observation(s.residents[0])
    assert all(m['value_gap']==m['dividend_yield_today']==0 for m in obs['market'])
    assert all(m['firm_cash'] is None for m in obs['market'])


def test_replay_is_exactly_deterministic_with_persistent_personal_states():
    a,b=RetailReplay.create(30,42),RetailReplay.create(30,42)
    for i in range(30):
        prices=[1200+i,1000-i,900+i*2]
        assert a.emit(prices)==b.emit(prices)
    assert [x.cash for x in a.residents]==[x.cash for x in b.residents]
    assert any(x.memories[-1]['event'] in ['sale','purchase'] for x in a.residents)


def test_ridge_standardization_fits_training_data_only():
    x=np.arange(100)/100
    frame=pd.DataFrame({'code':['a']*100,'x':x,'y':2*x+1})
    fit=fit_linear(frame,'y',['x'],alpha=.0001)
    before=copy.deepcopy(fit)
    test=pd.DataFrame({'code':['a'],'x':[1000]})
    p=predict_linear(fit,test)
    assert fit==before and fit['mean'][0]==pytest.approx(.495)
    assert p[0]==pytest.approx(2001,rel=.001)


def test_intercept_and_zero_baselines_have_distinct_meaning():
    train=pd.DataFrame({'code':['a','a','b','b'],'flow_proxy':[1,3,-1,-3]})
    test=pd.DataFrame({'code':['a','b','new']})
    assert predict_engine(fit_engine(train,'flow_proxy','train_mean'),test).tolist()==[2,-2,0]
    assert predict_engine(fit_engine(train,'flow_proxy','zero'),test).tolist()==[0,0,0]


def test_paired_bootstrap_keeps_cross_section_together():
    dates=np.repeat(pd.bdate_range('2025-01-01',periods=50).strftime('%Y-%m-%d'),2)
    f=pd.DataFrame({'label_date':dates,'target':1.,'baseline':0.,'candidate':.8})
    result=paired_block_interval(f,'target','candidate','baseline',5,draws=200)
    assert result['dates']==50 and result['improvement_supported']
    assert result['lower']==pytest.approx(.96) and result['upper']==pytest.approx(.96)


def test_oos_r2_compares_to_frozen_training_mean_not_test_mean():
    m=predictive_metrics([1,1],[0,0],[.5,.5])
    assert m['oos_r2_vs_training_stock_mean']==-3
    assert m['correlation'] is None


def test_completed_research_runs_cannot_be_overwritten(tmp_path):
    (tmp_path/'manifest.json').write_text('{"source_sha256":{}}')
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='immutable'):
        run_analysis(tmp_path)


def test_constant_series_has_no_spurious_autocorrelation():
    assert autocorrelation([1]*50)==0


def test_persistent_adjustment_basis_revision_is_rejected():
    _,m=data_fixture(50);old=m['005930']['price'];new=old.copy();new['close']*=.77
    assert quote_basis_check(old,new)['suspect_adjustment_basis']
    new=old.copy();new.loc[49,'close']*=.98
    assert not quote_basis_check(old,new)['suspect_adjustment_basis']
