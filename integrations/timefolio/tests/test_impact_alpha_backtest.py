import copy

import numpy as np
import pandas as pd
import pytest

from quant.impact_alpha_backtest import extract_signals, fill_price, quote_at, rule_mask, simulate


def market():
    times = np.arange(0,60001,100,dtype=np.int64)
    books = np.zeros((len(times),2,5,2))
    for level in range(5):
        books[:,0,level,0] = 99.99-level*.01
        books[:,1,level,0] = 100.01+level*.01
        books[:,:,level,1] = 10
    trades = pd.DataFrame([(2000,1,'Buy',2.,100.01)],
                         columns=['ts','connection','side','quantity','price'])
    return {'times':times,'books':books,'epochs':np.ones(len(times),np.int64),'trades':trades}


def test_features_use_one_second_only_even_if_no_future_book_exists():
    data = market()
    full,_ = extract_signals(data,100,60000)
    clipped = {**data, **{k:data[k][:31] for k in ['times','books','epochs']}}
    past,_ = extract_signals(clipped,100,60000)
    pd.testing.assert_frame_equal(full,past)
    assert full.iloc[0].decision_ms == 3000
    assert full.iloc[0].feature_book_max_ms <= 3000


def test_future_price_and_refill_cannot_change_already_decided_signal():
    data = market(); original,_ = extract_signals(data,100,60000)
    data['books'][31:,:,:,0] += 100
    data['books'][31:,:,:,1] += 999
    altered,_ = extract_signals(data,100,60000)
    pd.testing.assert_frame_equal(original,altered)


def test_past_refill_can_change_rule_without_using_later_refill():
    data = market()
    data['books'][21:,1,0,1] = 8
    data['books'][25:,1,0,1] = 12
    signals,_ = extract_signals(data,100,60000)
    assert signals.iloc[0].refill_ratio == pytest.approx(2)
    assert not rule_mask(signals,'breakout',100,200).any()


def test_entry_waits_for_refill_observation_plus_latency_and_pays_spread():
    data = market(); signals,_ = extract_signals(data,100,60000)
    rows,_ = simulate(data,signals,'breakout',5,100,200,'2026-09-01')
    row = rows.iloc[0]
    assert row.entry_ms == 3100 and row.exit_ms == 8100
    assert row.entry_price == 100.01 and row.exit_price == 99.99
    assert row.entry_quote_ms <= row.entry_ms
    assert row.exit_quote_ms <= row.exit_ms


def test_vwap_walks_book_and_cannot_invent_liquidity():
    levels=np.array([[100.,.05],[102.,.05]])
    before=levels.copy()
    assert fill_price(levels,'buy',.1) == pytest.approx(101)
    assert fill_price(levels,'buy',.2) is None
    np.testing.assert_array_equal(before,levels)


def test_missing_exit_fails_instead_of_dropping_an_open_trade():
    data=market(); signals,_=extract_signals(data,100,60000)
    keep=(data['times']<4000)|(data['times']>15000)
    for k in ['times','books','epochs']:data[k]=data[k][keep]
    with pytest.raises(ValueError,match='Unpriced open position'):
        simulate(data,signals,'breakout',5,100,200,'2026-09-01')


def test_delayed_exit_is_recorded_and_position_overlap_is_suppressed():
    data=market(); signals,_=extract_signals(data,100,60000)
    second=signals.iloc[0].copy();second['event_ms']=4000;second['decision_ms']=5000
    signals=pd.concat([signals,pd.DataFrame([second])],ignore_index=True)
    keep=(data['times']<7000)|(data['times']>=9000)
    for k in ['times','books','epochs']:data[k]=data[k][keep]
    rows,audit=simulate(data,signals,'breakout',5,100,200,'2026-09-01')
    assert len(rows)==1 and rows.iloc[0].exit_ms==9000
    assert rows.iloc[0].exit_delay_ms==900
    assert audit['position_already_open']==1


def test_known_bad_interval_cannot_carry_forward_a_fresh_looking_quote():
    data=market();data['invalid_intervals']=[{'start_ms':2990,'end_ms':3050}]
    assert quote_at(data,3000) is None
    signals,audit=extract_signals(data,100,60000)
    assert signals.empty and audit['excluded']['past_gap']==1
