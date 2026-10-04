"""Research checks: point-in-time boundaries, units, labels and real holdout."""
from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from research import news_swarm as s


def event():
    return {"id": "005930_2026-09-01_historical", "code": "005930", "company": "삼성전자",
            "mode": "historical", "cutoff_at": "2026-09-01T16:00:00+09:00",
            "documents": [{"kind": "news", "title": "삼성전자 신규 계약", "summary": "내용"}],
            "price_return": .9, "individual_shares": 123, "account": "private"}


def market():
    dates = pd.bdate_range("2026-09-01", periods=7).strftime("%Y-%m-%d")
    prices = pd.DataFrame({"date": dates, "open": [100, 110, 112, 114, 116, 118, 120],
                           "close": [105, 111, 113, 115, 117, 119, 121], "volume": 1000})
    flows = pd.DataFrame({"date": dates, "institution_shares": 20, "foreign_shares": -50,
                          "individual_shares": 25, "residual_shares": 30})
    return prices, flows


def record(mode="historical"):
    return {"mode": mode, "cutoff_at": "2026-09-01T16:00:00+09:00",
            "predicted_at": "2026-09-02T16:01:00+09:00"}


def test_loopback_utc_and_collector_timestamps_agree():
    assert s.timestamp("2026-09-01 16:00:00") == s.timestamp("2026-09-01T07:00:00Z")
    with pytest.raises(ValueError):
        s.timestamp("")


def test_model_inputs_cannot_include_evaluation_or_account_data():
    payload = json.dumps(s.input_batch([event()]), ensure_ascii=False)
    assert "price_return" not in payload
    assert "individual_shares" not in payload
    assert "account" not in payload
    assert "신규 계약" in payload


def test_batching_never_exposes_later_events_to_earlier_predictions():
    first = event()
    same_time = {**event(), "id": "000660", "code": "000660"}
    later = {**event(), "id": "later", "cutoff_at": "2026-09-02T16:00:00+09:00"}
    batches = s.causal_batches([first, later, same_time], 8)
    assert len(batches) == 2
    assert [x["id"] for x in batches[0]] == [first["id"], "000660"]
    assert all(len({e["cutoff_at"] for e in b}) == 1 for b in batches)


def test_invalidated_and_old_version_runs_cannot_be_reused(tmp_path):
    s.write_json(tmp_path / "excluded_from_validation.json", {"reason": "future leakage"})
    with pytest.raises(ValueError, match="excluded"):
        s.check_run(tmp_path)
    (tmp_path / "excluded_from_validation.json").unlink()
    s.write_json(tmp_path / "manifest.json", {"version": "old"})
    with pytest.raises(ValueError, match="version"):
        s.check_run(tmp_path)


def test_availability_filter_and_dedup_do_not_backdate_news():
    docs = [{"kind": "news", "title": "삼성전자 신규 계약", "summary": "", "available_at": t}
            for t in ["2026-09-01T15:00:00+09:00", "2026-09-01T15:01:00+09:00",
                      "2026-09-01T16:01:00+09:00"]]
    docs.append({"kind": "news", "title": "다른 회사 신규 계약", "summary": "", "available_at": docs[0]["available_at"]})
    e = s.make_event("005930", s.timestamp("2026-09-01 16:00:00"), s.timestamp("2026-09-01 00:00:00"), docs, "historical")
    assert len(e["documents"]) == 1
    assert e["documents"][0]["available_at"].endswith("15:00:00+09:00")


def test_pressure_uses_cash_and_inventory_not_vote_counts():
    decisions = [{"action": "buy", "fraction": 1}, {"action": "sell", "fraction": 1},
                 {"action": "hold", "fraction": 0}, {"action": "hold", "fraction": 0}]
    assert s.pressure(decisions) == pytest.approx((70 - 50) / 400)
    with pytest.raises(ValueError):
        s.pressure(decisions[:3])


@pytest.mark.parametrize("invalid", [
    {"fraction": float("nan")}, {"fraction": 1.1}, {"fraction": True},
    {"action": "hold", "fraction": .5}, {"evidence": []},
    {"evidence": [2]}, {"evidence": [True]}, {"id": "unknown"},
])
def test_invalid_model_outputs_fail_instead_of_becoming_neutral(invalid):
    row = {"id": event()["id"], "action": "buy", "fraction": .5,
           "price_score": .3, "evidence": [0], **invalid}
    with pytest.raises(ValueError):
        s.validate_decisions({"decisions": [row]}, [event()])


def test_exact_five_session_horizon_does_not_shorten_when_not_mature():
    p, f = market()
    assert s.label(record(), p.iloc[:3], f, 5, s.timestamp("2026-10-01")) is None
    outcome = s.label(record(), p, f, 5, s.timestamp("2026-10-01"))
    assert outcome["label_end_date"] == "2026-09-08"
    assert outcome["residual_shares"] == 150
    assert outcome["individual_shares"] == 125
    assert outcome["residual_ratio"] == pytest.approx(150 / 5000)
    assert outcome["price_return"] == pytest.approx(119 / 105 - 1)
    assert outcome["entry_return"] == pytest.approx(119 / 110 - 1)


def test_prospective_uses_completion_time_and_next_open():
    p, f = market()
    outcome = s.label(record("prospective"), p, f, 1, s.timestamp("2026-10-01"))
    assert outcome["date"] == "2026-09-02"
    assert outcome["label_end_date"] == "2026-09-03"
    assert outcome["entry_return"] == pytest.approx(113 / 112 - 1)


def test_provisional_day_missing_flow_and_stale_reference_are_excluded():
    p, f = market()
    assert s.label(record(), p, f, 1, s.timestamp("2026-09-02 19:00:00")) is None
    f.loc[1, "individual_shares"] = np.nan
    assert s.label(record(), p, f, 1, s.timestamp("2026-10-01")) is None
    assert s.label(record(), p.iloc[1:], f, 1, s.timestamp("2026-10-01")) is None


def test_canonical_market_sessions_handle_weekend_and_reject_missing_stock_bars():
    p, f = market()
    sessions = list(p.date)
    weekend = {**record("prospective"), "predicted_at": "2026-09-05T12:00:00+09:00"}
    outcome = s.label(weekend, p, f, 1, s.timestamp("2026-10-01"), sessions)
    assert outcome["base_date"] == "2026-09-04"
    assert outcome["label_end_date"] == "2026-09-07"
    missing_next_day = p[p.date != "2026-09-02"]
    assert s.label(record(), missing_next_day, f, 1, s.timestamp("2026-10-01"), sessions) is None


def test_purged_holdout_excludes_training_labels_reaching_test_period():
    dates = pd.bdate_range("2026-01-01", periods=50)
    rng = np.random.default_rng(42)
    rows = []
    for pos, date in enumerate(dates):
        for code in range(3):
            score = float(rng.normal())
            rows.append({"date": date.strftime("%Y-%m-%d"), "code": code,
                         "label_end_date": (date + pd.offsets.BDay(5)).strftime("%Y-%m-%d"),
                         "score": score, "target": 2 * score + .01 * rng.normal()})
    frame = pd.DataFrame(rows)
    result = s.regress(frame, "target", "score", dates[35].strftime("%Y-%m-%d"), 5)
    assert result["train_n"] == 30 * 3
    assert result["test_n"] == 15 * 3
    assert result["beta"] == pytest.approx(2, abs=.02)
    assert result["oos"]["oos_r2_vs_train_mean"] > .99
    assert result["beta_pvalue_panel_hac"] is None


def test_direction_accuracy_reports_abstention_coverage():
    out = s.metrics(np.array([-1., 1., 1.]), np.array([-1., 0., 1.]), np.array([-1., 1., 1.]))
    assert out["direction_accuracy"] == 1
    assert out["direction_coverage"] == pytest.approx(2 / 3)


def test_public_provider_units_and_euc_kr_xml(monkeypatch):
    class Reply:
        text = '<?xml version="1.0" encoding="EUC-KR"?><root><item data="20260901|100|110|90|105|1000"/></root>'

        def raise_for_status(self):
            pass

        def json(self):
            return [{"bizdate": "20260901", "organPureBuyQuant": "-1,000",
                     "foreignerPureBuyQuant": "+300", "individualPureBuyQuant": "+690"}]

    monkeypatch.setattr(s.requests, "get", lambda *a, **k: Reply())
    prices, flows = s.fetch_market("005930")
    assert prices.iloc[0].close == 105
    assert prices.iloc[0].volume == 1000
    assert flows.iloc[0].residual_shares == 700
    assert flows.iloc[0].individual_shares == 690


def test_refresh_preserves_old_labels_and_backs_up_before_upsert(tmp_path, monkeypatch):
    prices, flows = market()
    prices.to_csv(tmp_path / "prices_005930.csv", index=False)
    flows.to_csv(tmp_path / "flows_005930.csv", index=False)
    incoming_prices, incoming_flows = prices.iloc[2:].copy(), flows.iloc[2:].copy()
    incoming_flows.loc[2, "individual_shares"] = 28
    monkeypatch.setattr(s, "fetch_market", lambda code: (incoming_prices, incoming_flows))
    s.snapshot_market(tmp_path, ["005930"])
    stored = pd.read_csv(tmp_path / "flows_005930.csv")
    assert len(stored) == len(flows)
    assert stored.iloc[0].individual_shares == 25
    assert stored.iloc[2].individual_shares == 28
    copies = list((tmp_path / "market_history").rglob("previous_flows_005930.csv"))
    assert len(copies) == 1
    assert pd.read_csv(copies[0]).iloc[2].individual_shares == 25


def test_format_repair_keeps_invalid_reply_for_audit(tmp_path, monkeypatch):
    base = {"id": event()["id"], "action": "buy", "fraction": .2,
            "price_score": .1, "evidence": [0]}
    replies = iter([{"result": {"decisions": [{**base, "evidence": [1]}]}, "usage": {}},
                    {"result": {"decisions": [base]}, "usage": {}}])
    monkeypatch.setattr(s, "llm_json", lambda *a, **k: next(replies))
    cache = tmp_path / "call.json"
    assert s.decision_call(cache, s.SYSTEM, {}, [event()])[event()["id"]]["evidence"] == [0]
    assert (tmp_path / "call.invalid0.json").exists()
    assert cache.exists()


def test_research_rejects_nonlocal_model_before_http(monkeypatch):
    import config
    monkeypatch.setattr(config, "LOCAL_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setattr(s.requests, "post", lambda *a, **k: pytest.fail("External LLM request"))
    with pytest.raises(ValueError, match="loopback"):
        s.llm_json(s.SYSTEM, {})


def test_personas_interact_only_after_all_independent_votes_and_resume_is_idempotent(tmp_path, monkeypatch):
    e = event()
    s.write_json(tmp_path / "events.json", [e])
    calls = []

    def complete(system, payload):
        calls.append(payload)
        row = {"id": e["id"], "price_score": .2, "evidence": [0], "reason": "근거"}
        row.update({"action": "buy", "fraction": .5} if "persona" in payload else {"flow_score": .2})
        return {"result": {"decisions": [row]}, "usage": {"completion_tokens": 10}}

    monkeypatch.setattr(s, "llm_json", complete)
    s.predict(tmp_path)
    assert len(calls) == 9
    assert all("peer_opinions" not in p for p in calls[:4])
    assert all(len(p["peer_opinions"][e["id"]]) == 3 for p in calls[4:8])
    s.predict(tmp_path)
    assert len(calls) == 9
    assert len(s.read_json(tmp_path / "predictions.json")) == 1


def test_end_to_end_evaluation_with_known_forward_signal_and_costs(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(19)
    scores = rng.uniform(-1, 1, 60)
    price_rows, flow_rows = [], []
    close = 100.0
    for i, day in enumerate(dates):
        previous_score = scores[i - 1] if i else 0
        op = close
        close = op * (1 + .01 * previous_score)
        price_rows.append({"date": day, "open": op, "close": close, "volume": 1000})
        flow = int(200 * previous_score)
        flow_rows.append({"date": day, "institution_shares": -flow, "foreign_shares": 0,
                          "individual_shares": flow, "residual_shares": flow})
    pd.DataFrame(price_rows).to_csv(tmp_path / "prices_005930.csv", index=False)
    pd.DataFrame(price_rows).to_csv(tmp_path / "prices_069500.csv", index=False)
    pd.DataFrame(flow_rows).to_csv(tmp_path / "flows_005930.csv", index=False)
    s.write_json(tmp_path / "manifest.json", {"mode": "historical", "codes": ["005930"], "limitations": []})
    records = []
    for day, score in zip(dates[:50], scores):
        records.append({"id": day, "code": "005930", "mode": "historical", "cutoff_at": day + "T16:00:00+09:00",
                        "predicted_at": "2026-06-01T16:00:00+09:00", **{
                            f"{engine}_{suffix}": float(score) for engine in ["baseline", "independent", "swarm"]
                            for suffix in ["flow", "price"]}})
    s.write_json(tmp_path / "predictions.json", records)
    monkeypatch.setattr(s, "now", lambda: s.timestamp("2026-06-30"))
    report = s.evaluate(tmp_path, cost_bps=20)
    assert report["predictions"] == 50
    assert report["matured_labels"] == 100
    assert report["flow_proxy_check"]["direction_agreement"] == 1
    fitted = report["regressions"]["entry_return_from_flow_1d"]["swarm"]
    assert fitted["beta"] == pytest.approx(.01)
    assert fitted["oos"]["oos_r2_vs_train_mean"] > .99
    profit = report["oos_long_only_from_flow"]["swarm"]
    assert profit["trades"] > 0
    assert profit["mean_return_per_trade_after_cost"] > 0
    assert report["verified_profitable"] is False
    assert (tmp_path / "report.json").exists()
