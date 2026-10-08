from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from arcmarket.systematic import Book, clean_bars, features, select
from quant.krx_cnn import image_windows, purged_masks
from quant.krx_cnn_paper import advance, eligible_execution


def bar(price=100, volume=1000):
    return {"open": price, "high": price, "low": price, "close": price, "volume": volume}


def test_self_financing_whole_shares_both_costs_and_no_churn():
    b = Book(cash=1000)
    b.step("2026-01-02", {"A": bar(10)}, weights={"A": .08}, previous={"A": 10}, adv={"A": 1e6})
    assert b.positions == {"A": 8}
    assert b.cash == pytest.approx(919.84)
    assert b.nav() == pytest.approx(999.68)
    b.step("2026-01-05", {"A": bar(10)}, weights={"A": .08}, previous={"A": 10}, adv={"A": 1e6})
    assert len(b.trades) == 1
    b.step("2026-01-06", {"A": bar(10)}, weights={}, previous={"A": 10}, adv={"A": 1e6})
    assert b.positions == {} and b.cash == pytest.approx(999.68)


def test_limit_up_no_buy_limit_down_no_sell_and_no_negative_cash():
    b = Book(cash=1000)
    b.step("2026-01-02", {"A": bar(13)}, weights={"A": .08}, previous={"A": 10}, adv={"A": 1e6})
    assert not b.trades
    b.positions = {"A": 8}; b.marks={"A": 10}
    b.step("2026-01-05", {"A": bar(7)}, weights={}, previous={"A": 10}, adv={"A": 1e6})
    assert b.positions["A"] == 8 and b.cash >= 0


def test_halt_uses_observed_close_without_fabricating_a_fill():
    d = clean_bars(pd.DataFrame([dict(date="2026-01-02", open=0, high=0, low=0, close=100, volume=0)]))
    assert d.close.iloc[0] == 100 and pd.isna(d.open.iloc[0])
    b = Book(cash=0, positions={"A": 2}, marks={"A": 90})
    b.step("2026-01-02", {"A": d.iloc[0].to_dict()}, weights={}, previous={"A": 90}, adv={"A": 1e9})
    assert b.positions == {"A": 2} and not b.missing and not b.trades
    assert b.nav() == pytest.approx(199.6)
    b.step("2026-01-05", {})
    assert b.missing == {"A": 1}  # unknown price is never zero or silently dropped


def test_future_changes_cannot_change_historical_features_or_image():
    idx=pd.bdate_range("2025-01-01", periods=90)
    d=pd.DataFrame([bar(100+i, 1e8) for i in range(90)], index=idx)
    changed=d.copy(); changed.loc[idx[75]:, ["open","high","low","close"]]*=2
    a=features({"A": d}); b=features({"A": changed})
    pd.testing.assert_frame_equal(a[a.date<idx[75]], b[b.date<idx[75]])
    x=image_windows([d.iloc[:60].to_numpy()]); y=image_windows([changed.iloc[:60].to_numpy()])
    assert np.array_equal(x,y)
    scaled=d.iloc[:60].to_numpy(); scaled[:,:4]*=3; scaled[:,4]*=4
    assert np.array_equal(x,image_windows([scaled]))


def test_label_horizon_is_purged_at_both_boundaries():
    m=pd.DataFrame({"date":pd.to_datetime(["2025-05-29","2025-05-01","2025-08-29","2025-08-01","2025-09-01"]),
                    "label_end":pd.to_datetime(["2025-06-05","2025-05-08","2025-09-05","2025-08-08","2025-09-08"]),
                    "fwd":[.1]*5})
    tr,va,te=purged_masks(m)
    assert tr.tolist()==[False,True,False,False,False]
    assert va.tolist()==[False,False,False,True,False]
    assert te.tolist()==[False,False,False,False,True]


def test_paper_cannot_backdate_orders_and_repeat_does_not_refill():
    decision={"created_at":"2026-09-16T09:15:00+09:00","as_of":"2026-09-15",
              "weights":{"A":.08},"previous":{"A":100},"adv":{"A":1e9}}
    assert not eligible_execution("2026-09-16",decision)
    assert eligible_execution("2026-09-17",decision)
    state={"last_processed":"2026-09-15","accounts":{"cnn":{"book":asdict(Book()),"curve":[],"pending":decision}}}
    d=pd.DataFrame([bar(),bar()],index=pd.to_datetime(["2026-09-16","2026-09-17"]))
    advance(state,{"A":d},now="2026-09-18")
    assert len(state["accounts"]["cnn"]["book"]["trades"]) == 1
    assert state["accounts"]["cnn"]["book"]["trades"][0]["date"]=="2026-09-17"
    advance(state,{"A":d},now="2026-09-18")
    assert len(state["accounts"]["cnn"]["book"]["trades"]) == 1


def test_rank_is_bounded_and_missing_cnn_scores_do_not_fall_back():
    d=pd.DataFrame([dict(code=str(i),valid=True,adv=1e10,sigma=.02,r20=.1+i*.001,r60=.2+i*.001,r5=.01,
                          trend=True,close=100) for i in range(40)])
    p=select(d)
    assert len(p["weights"])==10 and sum(p["weights"].values())==pytest.approx(.8)
    assert select(d,scores={})["weights"]=={}
    assert select(d,scores={"39":.6})["weights"]=={"39":.08}
