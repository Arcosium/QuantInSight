import numpy as np
import pandas as pd
from autofolio.market_runner import price_features,baseline_signals
from autofolio.period import expected_dates


def sample():
    rows=[]
    for symbol,rate in [('a',.001),('b',-.001)]:
        for n,day in enumerate(pd.bdate_range('2023-01-02',periods=180)):
            price=100*np.exp(rate*n+.01*np.sin(n/4))
            rows.append(dict(date=day,symbol=symbol,open=price,high=price*1.01,low=price*.99,
                close=price,volume=1000,eligible=True,sector='UNKNOWN'))
    return pd.DataFrame(rows)


def test_future_changes_do_not_change_prior_signals():
    frame=sample();cut=pd.Timestamp('2023-07-03')
    changed=frame.copy();changed.loc[changed.date>cut,['open','high','low','close','volume']]*=3
    a=price_features(frame);b=price_features(changed)
    for representation in ('momentum','reversal','trend_momentum','lowvol_momentum'):
        g=dict(representation=representation,mode='regime_gate')
        left=baseline_signals(a,g);right=baseline_signals(b,g)
        pd.testing.assert_frame_equal(left[left.date<=cut],right[right.date<=cut])


def test_missing_quote_does_not_become_contiguous_history():
    frame=sample();frame=frame[~((frame.symbol=='a')&(frame.date==pd.Timestamp('2023-04-03')))]
    p=price_features(frame)
    assert not p.loc[(p.symbol=='a')&(p.date==pd.Timestamp('2023-04-04')),'history_ready'].iloc[0]
    assert p.loc[(p.symbol=='b')&(p.date==pd.Timestamp('2023-04-04')),'history_ready'].iloc[0]


def test_kr_announced_2026_closures_are_not_missing_quotes():
    days=expected_dates('kr','20260101','20261231')
    assert '20260603' not in days and '20260717' not in days
    assert '20260602' in days and '20260716' in days
