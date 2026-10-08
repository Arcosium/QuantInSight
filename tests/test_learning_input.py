import pandas as pd
import pytest
from autofolio.learning_input import validate_minutes


def test_minutes_order_and_boundary_duplicates():
    good=pd.DataFrame(dict(ts=[1000,2000],open=[1,1],high=[2,2],low=[.5,.5],close=[1,1],quote_volume=[1,1]))
    assert validate_minutes(good)==2000
    with pytest.raises(ValueError):validate_minutes(good.iloc[::-1])
    with pytest.raises(ValueError):validate_minutes(good,1000)
    bad=good.copy();bad.loc[1,'ts']=1000
    with pytest.raises(ValueError):validate_minutes(bad)
    bad=good.copy();bad.loc[0,'quote_volume']=-1
    with pytest.raises(ValueError):validate_minutes(bad)


def test_history_overrides_live_and_duplicate_rows_are_not_double_counted():
    from autofolio.learning_input import canonical_minutes
    hist=pd.DataFrame(dict(ts=[2000,1000],open=[1.,1.],high=[2.,2.],low=[.5,.5],close=[1.,1.],quote_volume=[2.,3.],_source=[0,0]))
    live=hist.copy();live['_source']=1;live['close']=1.5
    d,duplicates=canonical_minutes([live,hist])
    assert duplicates==2
    assert d.ts.tolist()==[1000,2000]
    assert d.close.tolist()==[1.,1.]
    assert d.quote_volume.sum()==5


def test_invalid_minutes_dropped_without_filling_and_coverage_uses_valid_count():
    import numpy as np
    from autofolio.learning_input import canonical_minutes,daily_minutes
    n=1201
    raw=pd.DataFrame(dict(ts=pd.Timestamp('2026-09-01').value//10**6+np.arange(n)*60000,
                          open=np.ones(n),high=np.ones(n)*2,low=np.ones(n)*.5,
                          close=np.ones(n),quote_volume=np.ones(n),_source=np.zeros(n)))
    raw.loc[0,'quote_volume']=np.nan
    raw.loc[1,'open']=0
    cleaned,_=canonical_minutes([raw],reject_invalid=True)
    assert cleaned.attrs['rejected_rows']==2
    day=daily_minutes(cleaned).iloc[0]
    assert day.minutes==1199
    assert day.volume==1199
    assert day.minutes<1200  # This day cannot satisfy the research eligibility gate.
    raw.loc[2,'high']=-1
    cleaned,_=canonical_minutes([raw],reject_invalid=True)
    assert cleaned.attrs['rejected_rows']==3


def test_cache_checks_source_signature_and_derived_content(tmp_path,monkeypatch):
    from autofolio import learning_input
    monkeypatch.setattr(learning_input,'RUNS',tmp_path)
    frame=pd.DataFrame({'date':[pd.Timestamp('2026-09-01')],'close':[1.]})
    signature=learning_input.source_signature([dict(path='source',size=10,mtime_ns=1)],'20261001')
    learning_input.save_symbol('BTC',frame,dict(source_signature=signature,rejected_rows={},duplicate_minutes=0))
    assert learning_input.cached_symbol('BTC',signature) is not None
    changed=learning_input.source_signature([dict(path='source',size=10,mtime_ns=2)],'20261001')
    assert learning_input.cached_symbol('BTC',changed) is None
    data,_=learning_input.symbol_cache_paths('BTC');data.write_bytes(b'corrupted')
    assert learning_input.cached_symbol('BTC',signature) is None
