"""Conservation, counterfactual boundaries, reconstruction and API isolation."""
from copy import deepcopy
import gzip
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from quant.market_impact import simulate, validate_book
from quant.impact_research import reconstruct, analyse, first_sustained, recovery_summary, _curve_stats
from web import impact_lab


@pytest.fixture
def book():
    return {"bids": [[99.0, 2.0], [98.0, 4.0]], "asks": [[101.0, 2.0], [102.0, 4.0]]}


def test_walk_book_cost_and_input_immutability(book):
    original = deepcopy(book)
    result = simulate(book, quantity=3, slices=3, horizon=3, half_time=None, fee_bps=10)
    assert book == original
    single = result["single"]
    assert single["vwap"] == pytest.approx(304/3)
    assert single["initial_impact_bps"] == pytest.approx(50)
    assert single["book_walk_bps"] == pytest.approx(100/3)
    assert single["total_cost_bps"] == pytest.approx(400/3+10*(304/3)/100)
    assert result["split_saving_bps"] == pytest.approx(0)
    assert single["book_at_end"]["asks"] == [[101, 0], [102, 3]]


def test_sell_sign_and_spread_cost_are_symmetric(book):
    result = simulate(book, side="sell", quantity=3, slices=3, horizon=3)
    assert result["single"]["vwap"] == pytest.approx(296/3)
    assert result["single"]["initial_impact_bps"] == pytest.approx(50)
    assert result["single"]["cost_bps"] > 0


def test_no_invented_depth_or_cost_comparison_on_partial_fills(book):
    result = simulate(book, quantity=10, slices=5, horizon=5, half_time=.2)
    assert result["single"]["filled"] == 6
    assert result["single"]["unfilled"] == 4
    assert result["single"]["initial_impact_bps"] is None
    assert result["split"]["filled"] > result["single"]["filled"]
    assert result["split_saving_bps"] is None
    assert result["cost_comparable"] is False


def test_replenishment_preserves_caps_and_orders_execute_at_deadline(book):
    result = simulate(book, quantity=3, slices=5, horizon=2, half_time=.5)
    assert result["split"]["executions"][-1]["at_seconds"] == 2
    assert result["split"]["filled"] == pytest.approx(3)
    assert result["split_saving_bps"] > 0
    for side in ["asks", "bids"]:
        for current, initial in zip(result["split"]["book_at_end"][side], book[side]):
            assert 0 <= current[1] <= initial[1]


@pytest.mark.parametrize("kwargs", [{"quantity": float("nan")}, {"quantity": -1},
                                  {"quantity": 1e-14},
                                  {"half_time": 0}, {"side": "other"}, {"slices": 2.5},
                                  {"horizon": float("inf")}, {"fee_bps": -1}])
def test_invalid_experiment_rejected(book, kwargs):
    with pytest.raises(ValueError):
        simulate(book, **kwargs)


def test_invalid_duplicate_and_crossed_quotes_rejected():
    for book in [{"bids": [[102, 1]], "asks": [[101, 1]]},
                 {"bids": [[99, 1], [99, 2]], "asks": [[101, 1]]}]:
        with pytest.raises(ValueError):
            validate_book(book)


def test_reconstruction_absolute_updates_deletion_reset_dedup(tmp_path):
    bids = [[str(99-i), "2"] for i in range(50)]
    asks = [[str(101+i), "3"] for i in range(50)]
    messages = [
        {"topic": "orderbook.50.X", "type": "snapshot", "cts": 1000,
         "data": {"b": bids, "a": asks, "u": 100}},
        {"topic": "orderbook.50.X", "type": "delta", "cts": 1020,
         "data": {"b": [["99", "5"]], "a": [["101", "0"], ["151", "7"]], "u": 101}},
        {"topic": "orderbook.50.X", "type": "delta", "cts": 1020,
         "data": {"b": [["99", "99"]], "a": [], "u": 101}},
        {"topic": "publicTrade.X", "data": [{"i": "trade-1", "T": 1020, "S": "Buy", "v": "1", "p": "101"}]*2},
        {"topic": "orderbook.50.X", "type": "snapshot", "cts": 1030,
         "data": {"b": bids, "a": asks, "u": 1}},
    ]
    path = tmp_path/"raw.gz"
    with gzip.open(path, "wt") as f:
        for msg in messages:
            f.write(json.dumps({"segment": 1, "message": msg})+"\n")
    data = reconstruct(path)
    assert data["books"][1, 0, 0, 1] == 5  # replacement, not 2+5
    assert data["books"][1, 1, 0, 0] == 102
    assert data["books"][2, 0, 0, 1] == 2
    assert data["epochs"][0] != data["epochs"][-1]
    assert len(data["trades"]) == 1
    assert data["quality"]["duplicate_trades"] == 1


def test_censoring_not_mislabeled_as_zero_or_recovered_only_median():
    result = recovery_summary([.2, .5, None, None, None], 5)
    assert result["median_seconds"] is None
    assert result["censored"] == 3
    assert first_sustained([.1, .9, .1, .9, .91, .95, .97], .9) == .4
    assert first_sustained([1, 1, 1], .9) is None


def test_cluster_bootstrap_respects_cluster_count():
    result = _curve_stats([[1, 2], [1, 2], [2, 3]], groups=[1, 1, 2])
    assert result["clusters"] == 2
    assert result["mean"] == pytest.approx([4/3, 7/3])
    assert np.all(np.asarray(result["lower"]) <= result["mean"])


def test_api_requires_calibration_and_has_no_order_path(book, tmp_path, monkeypatch):
    monkeypatch.setattr(impact_lab, "ROOT", tmp_path)
    client = TestClient(impact_lab.preview_app)
    assert client.get("/api/impact/study").status_code == 503
    study = {"schema_version": 1, "symbol": "ETHUSDT",
             "snapshots": [{"id": "thin-buy", "regime": "thin", "event_side": "buy", "timestamp_ms": 1, **book}],
             "calibration": {"thin": {"empirical": False, "half_time_seconds": None}}}
    (tmp_path/"study.json").write_text(json.dumps(study))
    payload = {"snapshot_id": "thin-buy", "refill": "empirical", "quantity": 6}
    assert client.post("/api/impact/simulate", json=payload).status_code == 422
    payload["refill"] = "none"
    response = client.post("/api/impact/simulate", json=payload)
    assert response.status_code == 200
    assert response.json()["single"]["filled"] == 6
    payload["side"] = "sell"
    assert client.post("/api/impact/simulate", json=payload).status_code == 422
    assert client.post("/api/order", json={}).status_code == 404
    assert (tmp_path/"study.json").read_text() == json.dumps(study)


def _aligned_data():
    import pandas as pd
    times=np.arange(0,100001,100)
    books=np.zeros((len(times),2,50,2))
    mid=100+times/1e6
    for i in range(50):
        books[:,0,i,0]=mid-.01-i*.01
        books[:,1,i,0]=mid+.01+i*.01
        books[:,:,i,1]=(1+times[:,None]/1e5)
    records=[(t,1,"Buy" if i%2==0 else "Sell",10.,100.)
             for i,t in enumerate([5000,17000,29000,41000,53000,65000,77000])]
    return {"times":times,"books":books,"epochs":np.ones(len(times)),
            "connections":np.ones(len(times)),"quality":{},
            "trades":pd.DataFrame(records,columns=["ts","connection","side","quantity","price"])}


def test_event_alignment_is_strictly_before_and_missing_recovery_is_not_fabricated():
    result,events=analyse(_aligned_data(),{"symbol":"TESTUSDT"})
    assert len(events)==7
    assert events[0]["impact"][0] == pytest.approx((100.0051-100.0049)/100.0049*1e4)
    assert all(b["ts"]-a["ts"]>=11000 for a,b in zip(events,events[1:]))
    assert result["groups"]["all"]["t90"]["n"] == 0
    assert result["calibration"]["all"]["empirical"] is False
    json.dumps(result,allow_nan=False)


def test_event_crossing_reset_is_excluded_from_entire_followup():
    data=_aligned_data()
    data["epochs"][data["times"]>=47000]=2
    result,events=analyse(data,{"symbol":"TESTUSDT"})
    assert result["exclusions"]["connection_or_snapshot_boundary"]==1
    assert 41000 not in [e["ts"] for e in events]
