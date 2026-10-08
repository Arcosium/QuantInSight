import numpy as np
import pandas as pd
import pytest
from autofolio import research_input as r


def prices(n=130):
    return pd.DataFrame(dict(date=pd.bdate_range('2023-01-02',periods=n),open=np.ones(n)*10,high=np.ones(n)*11,low=np.ones(n)*9,close=np.ones(n)*10,volume=np.ones(n)*100))


def test_bridge_only_stable_basis():
    a=prices();c=a.copy();c[r.OHLC]*=2;c.volume*=.5
    _, audit=r.bridge(c,a)
    assert audit['price_factor']==2
    assert audit['volume_factor']==.5
    c.loc[10,'close']=19
    with pytest.raises(ValueError,match='ambiguous_price'):r.bridge(c,a)


def test_volume_scope_mismatch_rejected():
    a=prices();c=a.copy();c.loc[5,'volume']=120
    with pytest.raises(ValueError,match='ambiguous_volume'):r.bridge(c,a)


def test_duplicate_and_invalid_ohlc_rejected():
    a=prices()
    with pytest.raises(ValueError,match='duplicate'):r.validate_frame(pd.concat([a,a.iloc[:1]]))
    a.loc[0,'high']=8
    with pytest.raises(ValueError,match='invalid_ohlcv'):r.validate_frame(a)


def test_missing_middle_day_not_ready(monkeypatch):
    monkeypatch.setattr(r,'expected_dates',lambda *args:('20230102','20230103','20230104'))
    a=prices(3);a['symbol']='A';a['eligible']=True
    assert r.coverage(a,'kr',minimum=1)['ready']
    result=r.coverage(a.drop(index=1),'kr',minimum=1)
    assert not result['ready'] and result['insufficient_dates']==['20230103']
