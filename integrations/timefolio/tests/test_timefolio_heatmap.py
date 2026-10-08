from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant.timefolio_heatmap_data import aggregate_minutes
from quant.timefolio_heatmap_features import labels, make_image, restore_price_basis, infer_actions
from quant.timefolio_heatmap_replay import replay, targets
from quant.timefolio_heatmap_study import configurations, purged_masks, transform
from quant.timefolio_heatmap_rule_amendment import designation_mask


def synthetic_panel(days=35, names=12):
    shape = (names, days)
    p = {k: np.ones(shape, dtype=float) for k in [
        "close", "o", "c", "v", "regular", "exec_price", "exec_volume", "exec_high", "exec_low", "exec_count",
        "market_cap", "listed_shares", "split", "factor", "adv5", "adv20", "v20", "r1", "r5", "r20", "vol20",
        "rank_adv", "rank_r5", "rank_vol", "sector_r5", "market_r5", "sector_cap"]}
    for k in ["close", "o", "c", "exec_price", "exec_high", "exec_low"]: p[k][:] = 10000
    p["exec_high"][:] = 10050; p["exec_low"][:] = 9950; p["exec_count"][:] = 30
    p["exec_volume"][:] = 1e7; p["v"][:] = 1e7; p["v20"][:] = 1e7
    p["market_cap"][:] = 2e12; p["adv5"][:] = 1e11; p["adv20"][:] = 1e11
    p["sector_cap"][:] = .1; p["sector"] = np.arange(names); p["eligible"] = np.ones(shape, bool)
    p["r1"][:] = .01; p["r5"][:] = .03; p["r20"][:] = .1; p["vol20"][:] = .02
    dates = pd.bdate_range("2026-04-01", periods=days).strftime("%Y%m%d").tolist()
    ix = {"dates": dates, "codes": [f"{i:06d}" for i in range(names)]}
    return p, ix


def test_aggregation_execution_is_after_signal_and_keeps_auction():
    times = pd.date_range("2026-04-01 09:00", "2026-04-01 15:19", freq="min").union(pd.DatetimeIndex(["2026-04-01 15:30"]))
    d = pd.DataFrame({"ts": times.strftime("%Y%m%d%H%M%S"), "o": 100., "h": 101., "l": 99., "c": 100., "v": 10.})
    bars, daily = aggregate_minutes(d)
    assert len(bars) == 26
    assert bars.iloc[-1]["count"] == 6
    assert daily.iloc[0].exec_volume == 300
    assert daily.iloc[0].exec_price == 100
    assert daily.iloc[0].regular
    assert daily.iloc[0]["count"] == 381


def test_purge_never_crosses_partition_and_whole_dates():
    dates = pd.bdate_range("2026-03-01", "2026-07-31").strftime("%Y%m%d").tolist()
    di = np.repeat(np.arange(len(dates)), 2); y = np.ones(len(di))
    train, val, refit, _, _ = purged_masks(dates, di, 10, y)
    dates = np.asarray(dates)
    assert np.all(dates[di[train] + 10] < "20260401")
    assert np.all(dates[di[val] + 10] < "20260701")
    assert np.all(dates[di[refit] + 10] < "20260701")
    assert np.all(train[::2] == train[1::2])


def test_target_rules_and_cash_reserve():
    scores = np.arange(20, 0, -1, dtype=float)
    sector = np.repeat(np.arange(10), 2)
    cap = np.repeat(.1, 20); mc = np.repeat(5e11, 20)
    w = targets(scores, np.ones(20, bool), sector, cap, mc, [f"{i:06}" for i in range(20)])
    assert w.sum() <= .3 + 1e-8
    for s in set(sector): assert w[sector == s].sum() <= .1 + 1e-8
    assert w.max() <= .15


def test_no_same_day_signal_fills_and_fee_accounting():
    p, ix = synthetic_panel()
    scores = np.full(p["close"].shape, np.nan)
    scores[:, 1:] = np.arange(12)[:, None]
    result = replay(p, ix, scores, ix["dates"][1], ix["dates"][8], rebalance=1, slip=0, return_trades=True)
    assert result["daily"][0]["holdings"] == 0
    assert result["daily"][1]["holdings"] > 0
    assert all(x["signal_date"] < x["date"] for x in result["trades"])
    assert min(x["cash"] for x in result["daily"]) >= 0
    assert result["daily"][-1]["nav"] == pytest.approx(1e9 - result["metrics"]["fees_krw"], abs=.01)


def test_volume_limits_and_no_short_positions():
    p, ix = synthetic_panel(); p["exec_volume"][:] = 100
    scores = np.ones(p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][5], participation=.01, return_trades=True)
    assert r["trades"]
    assert all(t["qty"] <= 1 for t in r["trades"])
    assert r["metrics"]["partial_fills"] > 0


def test_locked_upper_limit_prevents_buy():
    p, ix = synthetic_panel()
    p["exec_price"][:, 1] = p["exec_high"][:, 1] = p["exec_low"][:, 1] = 13000
    r = replay(p, ix, np.ones(p["close"].shape), ix["dates"][1], ix["dates"][1], return_trades=True)
    assert not r["trades"]
    assert r["metrics"]["blocked_fills"] > 0


def test_split_is_not_a_fifty_percent_loss():
    p, ix = synthetic_panel()
    p["split"][:, 3] = 2; p["factor"][:, 3:] = 2
    for k in ["close", "exec_price", "exec_high", "exec_low"]: p[k][:, 3:] /= 2
    r = replay(p, ix, np.ones(p["close"].shape), ix["dates"][1], ix["dates"][4], rebalance=5, slip=0)
    assert r["daily"][1]["nav"] == pytest.approx(r["daily"][2]["nav"])
    y, fwd = labels(p, np.array([0]), np.array([1]), 3, "absolute")
    assert fwd[0] == pytest.approx(0)


def test_future_changes_do_not_change_heatmap():
    p, _ = synthetic_panel(names=1)
    bars = np.ones((1, 35, 26, 6), np.float32)
    bars[..., :4] = 10000; bars[..., 4] = 100; bars[..., 5] = 15
    before = make_image(p, bars, 0, 25)
    future = deepcopy(p)
    for key, value in future.items():
        if value.ndim == 2 and value.dtype != bool: value[:, 26:] *= 7
        elif value.dtype == bool: value[:, 26:] = ~value[:, 26:]
    bars[:, 26:] *= 8
    np.testing.assert_array_equal(before, make_image(future, bars, 0, 25))


def test_layout_changes_preserve_information():
    a = np.arange(32 * 65, dtype=np.uint16).reshape(1, 32, 65) % 256
    cfg = configurations()[0]
    for layout in ["standard", "rules_top", "rules_middle", "interleave", "transpose", "rows_shuffled", "time_shuffled"]:
        out = transform(a, dict(cfg, layout=layout))
        np.testing.assert_array_equal(np.sort(a.ravel()), np.sort(out.ravel()))
    assert len(configurations()) == len({c["id"] for c in configurations()})


def test_vendor_adjustment_restores_prices_and_preserves_notional():
    bars = np.array([[[50., 52., 49., 51., 200., 15.]]])
    out = restore_price_basis(bars, [2.])
    assert out[0, 0, 3] == 102
    assert out[0, 0, 4] == 100
    assert out[0, 0, 3] * out[0, 0, 4] == bars[0, 0, 3] * bars[0, 0, 4]
    assert bars[0, 0, 3] == 51  # Original archive remains untouched.


def test_new_bonus_shares_cannot_be_sold_before_listing():
    p, ix = synthetic_panel()
    p["split"][:, 3] = 2; p["factor"][:, 3:] = 2
    for k in ["close", "exec_price", "exec_high", "exec_low"]: p[k][:, 3:] /= 2
    p["listed_shares"][:, 5:] *= 2
    scores = np.ones(p["close"].shape)
    # After the event all target budgets become zero, so the simulator wants to sell.
    p["sector_cap"][:, 2:] = 0
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][5], rebalance=1, slip=0, return_trades=True)
    before = {}
    for t in r["trades"]:
        if t["date"] < ix["dates"][3]:
            before[t["code"]] = before.get(t["code"], 0) + t["qty"] * (1 if t["side"] == "buy" else -1)
    # At the ex-date only the old shares may be sold; the bonus shares unlock later.
    ex_sales = [t for t in r["trades"] if t["side"]=="sell" and t["date"]==ix["dates"][3]]
    assert ex_sales
    assert all(t["qty"] <= before[t["code"]] for t in ex_sales)
    assert r["metrics"]["unreleased_rights_positions"] == 0


def test_truncated_session_does_not_create_and_reverse_an_action():
    basis = np.array([[1., .98, 1., 1.], [4., 1.016, 1., 1.]])
    regular = np.array([[True, False, True, True], [True, False, True, True]])
    closes = np.array([[100., 101., 100., 99.], [400., 110., 109., 108.]])
    actions = infer_actions(basis, regular, closes)
    np.testing.assert_array_equal(actions[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(actions[1], [1, 4, 1, 1])


def test_executed_buys_respect_sector_and_small_cap_limits_after_fees():
    p, ix = synthetic_panel(names=20)
    p["sector"] = np.repeat(np.arange(10), 2)
    p["market_cap"][:12] = 5e11
    scores = np.broadcast_to(np.arange(20, 0, -1)[:, None], p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][1],
               weight=.15, slip=.0025, return_trades=True)
    quantities = np.zeros(20); cash = 1e9
    for trade in r["trades"]:
        i = ix["codes"].index(trade["code"])
        assert trade["side"] == "buy"
        cash -= trade["qty"] * trade["price"] + trade["fee"]
        quantities[i] += trade["qty"]
        values = quantities * 10000; nav = cash + values.sum()
        assert cash >= 0
        assert values.max() / nav <= .15
        assert values.sum() / nav <= .8
        assert values[:12].sum() / nav <= .3
        for sector in np.unique(p["sector"]):
            assert values[p["sector"] == sector].sum() / nav <= .1


def test_price_driven_sector_breach_is_repaired_next_day_between_rebalances():
    p, ix = synthetic_panel(names=20)
    p["sector"] = np.repeat(np.arange(10), 2)
    for key in ["close", "exec_price", "exec_high", "exec_low"]:
        p[key][:2, 2:] *= 2
    scores = np.broadcast_to(np.arange(20, 0, -1)[:, None], p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][3],
               rebalance=99, slip=0, return_trades=True)
    assert r["metrics"]["closing_weight_breach_days"] == 1
    assert any(t["side"] == "sell" and t["date"] == ix["dates"][3] for t in r["trades"])
    quantities = np.zeros(20)
    for t in r["trades"]:
        quantities[ix["codes"].index(t["code"])] += t["qty"] * (1 if t["side"] == "buy" else -1)
    assert (quantities[:2] * p["close"][:2, 3]).sum() / r["daily"][-1]["nav"] <= .1


def test_designation_intervals_and_single_session_attention():
    dates = ["20260105", "20260106", "20260107", "20260108"]
    rows = [{"ISU_CD": "000001", "DESIGN_DD": "2026/01/06", "RELEAS_DD": "2026/01/08"}]
    np.testing.assert_array_equal(designation_mask(["000001"], dates, rows), [[False, True, True, False]])
    np.testing.assert_array_equal(designation_mask(["000001"], dates, rows, single_session=True), [[False, True, False, False]])


def test_designation_blocks_buy_on_effective_day_and_allows_sale():
    p, ix = synthetic_panel()
    p["trade_allowed"] = np.ones(p["eligible"].shape, bool)
    p["trade_allowed"][0, 2:] = False
    scores = np.broadcast_to(np.arange(12, 0, -1)[:, None], p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][4], rebalance=1, return_trades=True)
    assert any(t["code"] == "000000" and t["side"] == "buy" for t in r["trades"])
    assert any(t["code"] == "000000" and t["side"] == "sell" and t["date"] == ix["dates"][2] for t in r["trades"])
    assert not any(t["code"] == "000000" and t["side"] == "buy" and t["date"] >= ix["dates"][2] for t in r["trades"])


def test_individual_weight_drift_is_repaired_without_sector_breach():
    p, ix = synthetic_panel()
    p["sector_cap"][:] = .8
    for key in ["close", "exec_price", "exec_high", "exec_low"]:
        p[key][0, 2:] *= 3
    scores = np.broadcast_to(np.arange(12, 0, -1)[:, None], p["close"].shape)
    r = replay(p, ix, scores, ix["dates"][1], ix["dates"][3], rebalance=99, return_trades=True)
    assert r["metrics"]["closing_weight_breach_days"] == 1
    assert any(t["code"] == "000000" and t["side"] == "sell" and t["date"] == ix["dates"][3] for t in r["trades"])
