from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant.timefolio_heatmap_portfolio_selector import select_past


def fixture_returns():
    dates=pd.bdate_range('2026-01-02',periods=80).strftime('%Y%m%d').tolist()
    noise=np.tile([-.001,.001],40)
    returns={'a':.002+noise,'b':.0002+noise}
    return dates,returns


def test_origin_and_future_portfolio_returns_cannot_change_choice():
    dates,returns=fixture_returns();origin=dates[60]
    before=select_past(['a','b'],returns,dates,origin,20,1)
    changed=deepcopy(returns);changed['a'][60:]=-1.;changed['b'][60:]=10.
    assert select_past(['a','b'],changed,dates,origin,20,1)==before
    assert before[1]['last_observation']==dates[59]


def test_lookback_excludes_older_winner_and_uses_net_risk_adjusted_results():
    dates,returns=fixture_returns();returns['a'][40:60]=np.tile([-.02,.018],10)
    selected,meta=select_past(['a','b'],returns,dates,dates[60],20,1)
    assert selected==['b'];assert meta['observations']==20


def test_insufficient_history_uses_entire_pool_without_future_ranking():
    dates,returns=fixture_returns();returns['b'][10:]=100
    selected,meta=select_past(['a','b'],returns,dates,dates[10],60,1)
    assert selected==['a','b'];assert meta['observations']==10;assert 'fallback' in meta


def test_shadow_date_duplicates_are_rejected():
    dates,returns=fixture_returns();dates[20]=dates[19]
    with pytest.raises(ValueError,match='ordered'):
        select_past(['a','b'],returns,dates,dates[60],20,1)
