"""Synthetic-only validation; these tests never read the active live capture."""
from collections import Counter
import copy
from datetime import datetime, timezone
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from quant.impact_live_study import (analyse_live, block_stats, calibration_from_events,
                                      frozen_regimes, read_completed_source, select_snapshots,
                                      run, _legacy_pairing, interval_is_valid)


START = 1790553600000
REFERENCE = {"notional_threshold": 1677.4120724, "calibration_end": "2026-08-28",
             "regime_thresholds": {"depth": {"limits": [3.46863, 10.278992], "quantiles": [.2, .8]},
                                   "volatility": {"limits": [.063032, .190745], "quantiles": [1/3, 2/3]},
                                   "activity": {"limits": [.48479, 3.600414], "quantiles": [1/3, 2/3]}}}


def book(ms=0, update=1, kind="snapshot", connection=1):
    return {"segment": connection, "received_ns": (START+ms)*1000000,
            "message": {"topic": "orderbook.50.ETHUSDT", "type": kind,
                        "ts": START+ms+2, "cts": START+ms,
                        "data": {"s": "ETHUSDT", "u": update, "seq": update*10,
                                 "b": [[str(99.99-i*.01), "1"] for i in range(50)] if kind == "snapshot" else [],
                                 "a": [[str(100.01+i*.01), "1"] for i in range(50)] if kind == "snapshot" else []}}}


def trade(ms=50, identifier="a", connection=1):
    return {"segment": connection, "received_ns": (START+ms)*1000000,
            "message": {"topic": "publicTrade.ETHUSDT", "data": [
                {"i": identifier, "T": START+ms, "s": "ETHUSDT", "S": "Buy", "p": "100.01", "v": "20"}]}}


def source(root, rows=None, complete=True):
    root.mkdir(exist_ok=True)
    rows = rows if rows is not None else [book(), trade(), book(100, 2, "delta")]
    manifest = {"source": "Bybit public spot WebSocket", "symbol": "ETHUSDT", "depth": 50,
                "planned_seconds": 10, "elapsed_seconds": 10, "complete": complete,
                "started_utc": datetime.fromtimestamp(START/1000, timezone.utc).isoformat(),
                "ended_utc": datetime.fromtimestamp(START/1000+10, timezone.utc).isoformat(),
                "counts": dict(Counter(row["message"]["topic"] for row in rows)),
                "connections": max(row["segment"] for row in rows), "errors": []}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with gzip.open(root / "raw.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    return manifest


def synthetic_data(duration=720000, events=(15000, 16000, 26000, 38000, 350000, 361000, 700000)):
    times = START+np.arange(0, duration+1, 100)
    books = np.empty((len(times), 2, 50, 2), float)
    books[:, 0, :, 0] = 99.99-np.arange(50)*.01
    books[:, 1, :, 0] = 100.01+np.arange(50)*.01
    books[:, :, :, 1] = 1
    trades = pd.DataFrame([(START+t+3, 1, "Buy", 20., 100.01) for t in events],
                          columns=["ts", "connection", "side", "quantity", "price"])
    return {"times": times, "books": books, "epochs": np.ones(len(times), dtype=int),
            "connections": np.ones(len(times), dtype=int), "trades": trades, "quality": {}}


def test_incomplete_manifest_rejected_before_any_raw_open(tmp_path):
    (tmp_path / "manifest.json").write_text('{"complete":false}')
    with pytest.raises(ValueError, match="raw data was not opened"):
        read_completed_source(tmp_path)
    assert not (tmp_path / "raw.jsonl.gz").exists()


def test_completed_flag_cannot_hide_short_capture(tmp_path):
    manifest = source(tmp_path)
    manifest["elapsed_seconds"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "raw.jsonl.gz").unlink()
    with pytest.raises(ValueError, match="planned capture duration"):
        read_completed_source(tmp_path)


def test_crc_and_manifest_counts_are_verified(tmp_path):
    source(tmp_path)
    data, _, provenance = read_completed_source(tmp_path)
    assert provenance["gzip_crc_verified"] and provenance["message_counts_verified"]
    assert len(data["times"]) == 2
    raw = tmp_path / "raw.jsonl.gz"
    raw.write_bytes(raw.read_bytes()[:-8])
    with pytest.raises((OSError, EOFError)):
        read_completed_source(tmp_path)


def test_manifest_counts_and_supplied_hash_must_match(tmp_path):
    manifest = source(tmp_path)
    manifest["counts"]["orderbook.50.ETHUSDT"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="message counts"):
        read_completed_source(tmp_path)
    manifest = source(tmp_path)
    manifest["raw_sha256"] = "0"*64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256"):
        read_completed_source(tmp_path)


def test_nonunit_positive_update_increment_is_counted_not_assumed_missing(tmp_path):
    source(tmp_path, [book(), trade(), book(100, 4, "delta")])
    data, _, _ = read_completed_source(tmp_path)
    assert len(data["times"]) == 2
    assert data["epochs"][0] == data["epochs"][1]
    assert data["quality"]["nonunit_update_id_increments"] == 1


def test_bad_update_invalidates_until_new_snapshot_and_epochs_stay_separate(tmp_path):
    source(tmp_path, [book(), trade(), book(100, 2, "delta"), book(200, 1, "delta"),
                      book(300, 3, "delta"), book(400, 10), book(500, 11, "delta")])
    data, _, _ = read_completed_source(tmp_path)
    np.testing.assert_equal(data["times"], START+np.array([0, 100, 400, 500]))
    assert data["epochs"][1] != data["epochs"][2]
    assert data["quality"]["reordered_or_conflicting_duplicate_book"] == 1
    assert data["quality"]["delta_without_valid_snapshot"] == 1
    assert data["invalid_intervals"][0]["start_ms"] == START+100
    assert data["invalid_intervals"][0]["end_ms"] == START+400


@pytest.mark.parametrize("fault", ["cts", "seq", "connection", "crossed"])
def test_clock_sequence_connection_and_book_validation(tmp_path, fault):
    bad = book(100, 2, "delta")
    if fault == "cts":
        del bad["message"]["cts"]
    elif fault == "seq":
        bad["message"]["data"]["seq"] = 1
    elif fault == "connection":
        bad["segment"] = 2
    else:
        bad["message"]["data"]["b"] = [["101", "1"]]
    reset = book(200, 10, connection=2 if fault == "connection" else 1)
    source(tmp_path, [book(), trade(), bad, reset])
    data, _, _ = read_completed_source(tmp_path)
    np.testing.assert_equal(data["times"], START+np.array([0, 200]))


def test_identical_book_duplicate_is_dropped_without_breaking_epoch(tmp_path):
    update = book(100, 2, "delta")
    source(tmp_path, [book(), trade(), update, copy.deepcopy(update), book(200, 3, "delta")])
    data, _, _ = read_completed_source(tmp_path)
    assert len(data["times"]) == 3
    assert len(set(data["epochs"])) == 1
    assert data["quality"]["identical_duplicate_book_updates"] == 1


def test_regime_cutoffs_are_frozen_and_not_fitted_to_current_values():
    frame = pd.DataFrame({"depth": [1., 5., 50.], "volatility": [.01, .1, 1.],
                          "activity": [.1, 1., 10.], "spread_ticks": [1, 2, 3],
                          "directional_obi": [-.2, .2, 1.]})
    classified = frozen_regimes(frame, REFERENCE)
    assert list(classified.depth_regime) == ["Low", "Normal", "High"]
    assert list(classified.volatility_regime) == ["Low", "Normal", "High"]
    assert list(classified.obi_bin) == ["neutral", "positive", "positive"]
    alone = frozen_regimes(frame.iloc[[1]], REFERENCE)
    assert alone.iloc[0].depth_regime == "Normal"


def test_one_time_block_has_mean_but_no_inferential_interval():
    result = block_stats([[1., 3.], [3., 5.]], [123, 123])
    assert result["mean"] == [2., 4.]
    assert result["lower"] == [None, None]
    assert result["upper"] == [None, None]
    assert result["time_blocks"] == 1 and "days" not in result


def test_calibration_requires_eight_depletions_and_median_in_all_denominator():
    def event(value):
        return {"depleted": True, "regime": "thin", "t50_gap": value}
    assert not calibration_from_events([event(.5)]*7)["thin"]["empirical"]
    sparse = calibration_from_events([event(.5)]*3+[event(None)]*5)["thin"]
    assert not sparse["empirical"] and sparse["half_time_seconds"] is None
    assert sparse["censored"] == 5
    enough = calibration_from_events([event(.5)]*4+[event(None)]*4)["thin"]
    assert enough["empirical"] and enough["half_time_seconds"] == pytest.approx(.4)
    assert not calibration_from_events([event(.1)]*8)["thin"]["empirical"]


def test_snapshots_are_current_pre_event_medians_and_ignore_responses():
    data = synthetic_data(duration=5000, events=(1000,))
    frame = pd.DataFrame({"ts": START+np.array([1000, 2000, 3000]),
                          "depth": [1., 2., 3.], "depth_regime": ["Low"]*3,
                          "side": ["buy"]*3, "impact_300s": [100., -1000., 0.]})
    snapshots = select_snapshots(data, frame)
    assert len(snapshots) == 1
    assert snapshots[0]["id"] == "thin-buy"
    assert snapshots[0]["event_timestamp_ms"] == START+2000
    assert snapshots[0]["timestamp_ms"] == START+1900
    np.testing.assert_equal(snapshots[0]["asks"], data["books"][19, 1])


def test_full_synthetic_analysis_spacing_schema_and_empty_regime_safety():
    data = synthetic_data()
    result, frame, curves, post, legacy = analyse_live(data, {"symbol": "ETHUSDT"}, REFERENCE)
    assert np.diff(frame.ts).min() >= 11000
    long = frame[frame.long_primary]
    assert np.diff(long.ts).min() >= 301000
    assert result["schema_version"] == 1
    assert result["live_study"]["schema_version"] == 1
    assert result["live_study"]["cohorts"]["all"]["time_blocks"] == 2
    assert result["live_study"]["cohorts"]["second_hour"]["events"] == 0
    assert result["extension"]["regimes"]["depth"]["thresholds"] == REFERENCE["regime_thresholds"]["depth"]["limits"]
    assert result["groups"]["thin"]["n"] == 0
    assert not result["calibration"]["thin"]["empirical"]
    assert result["snapshots"] == []
    assert curves.shape == post.shape and curves.shape[0] == len(frame)
    json.dumps(result, allow_nan=False)


def test_zero_eligible_events_is_not_replaced_with_synthetic_observations():
    data = synthetic_data(duration=100000, events=(15000,))
    data["trades"]["quantity"] = .01
    result, frame, curves, post, legacy = analyse_live(data, {"symbol": "ETHUSDT"}, REFERENCE)
    assert result["live_study"]["events"] == 0
    assert frame.empty and len(legacy) == 0
    assert result["groups"]["all"]["n"] == 0 and result["snapshots"] == []
    assert result["extension"]["resiliency"]["early_impact_bps"] is None
    json.dumps(result, allow_nan=False)


def test_matching_cannot_cross_ten_minute_blocks():
    events = [{"ts": 599900, "minute": 0, "side": "buy", "quantity": 1.,
               "regime": "thin", "impact": np.ones(100)},
              {"ts": 600000, "minute": 1, "side": "buy", "quantity": 1.,
               "regime": "thick", "impact": np.zeros(100)}]
    assert _legacy_pairing(events, "regime", "thin", "thick")["pairs"] == 0
    events[0]["ts"], events[0]["minute"] = 600100, 1
    result = _legacy_pairing(events, "regime", "thin", "thick")
    assert result["pairs"] == 1
    assert result["difference_low_minus_high"]["lower"] == [None]*100


def test_stale_quotes_are_excluded_from_primary_without_reselecting_long_events():
    data = synthetic_data(duration=400000, events=(15000, 26000, 38000, 350000))
    keep = ~((data["times"] >= START+18000) & (data["times"] <= START+20100))
    for key in ["times", "books", "epochs", "connections"]:
        data[key] = data[key][keep]
    result, frame, *_ = analyse_live(data, {"symbol": "ETHUSDT"}, REFERENCE)
    first = frame[frame.ts == START+15000].iloc[0]
    second = frame[frame.ts == START+38000].iloc[0]
    assert not first.primary and first.long_primary
    assert second.primary and not second.long_primary
    # The 26 s event also sees the gap in its ten-second feature lookback.
    assert result["live_study"]["excluded_from_primary"]["quote_age_over_1000ms"] == 2


def test_run_exports_finite_json_provenance_and_calibration_audit_without_overwrite(tmp_path):
    capture = tmp_path / "synthetic_capture"
    source(capture)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REFERENCE))
    output = tmp_path / "result"
    result = run(capture, reference, output)
    assert result["provenance"]["source"]["manifest_complete"] is True
    assert len(result["provenance"]["reference_sha256"]) == 64
    assert set(["study.json", "live_study.json", "events.csv", "calibration_events.csv",
                "curves.npz", "run_manifest.json"]).issubset({p.name for p in output.iterdir()})
    columns = pd.read_csv(output / "calibration_events.csv").columns
    assert {"depleted", "t50_gap", "regime", "book_index", "ts", "depth"}.issubset(columns)
    assert "NaN" not in (output / "study.json").read_text()
    with pytest.raises(FileExistsError):
        run(capture, reference, output)


def test_known_invalid_interval_cannot_be_forward_carried_before_next_snapshot():
    data = synthetic_data(duration=720000, events=(15000, 400000))
    keep = ~((data["times"] >= START+100000) & (data["times"] < START+350000))
    for key in ["times", "books", "epochs", "connections"]:
        data[key] = data[key][keep]
    data["epochs"][data["times"] >= START+350000] = 2
    data["invalid_intervals"] = [{"start_ms": START+99900, "end_ms": START+350000,
                                  "reasons": ["bad_book"]}]
    result, frame, curves, *_ = analyse_live(data, {"symbol": "ETHUSDT"}, REFERENCE)
    first = frame[frame.ts == START+15000].iloc[0]
    assert first.primary  # Its 10-second window is before the invalid interval.
    assert not first.known_valid_300s and not first.long_primary
    assert np.isnan(first["impact_300s"])
    assert result["live_study"]["known_invalid_long_candidates"] == 1
    # The age-relaxed sensitivity cannot retain the original fabricated long path.
    assert result["live_study"]["cohorts"]["all_contiguous_sensitivity"]["resiliency"]["cohort_n"] == 1
    json.dumps(result, allow_nan=False)


def test_interval_guard_covers_lookback_and_open_ended_invalid_spans():
    invalid = [{"start_ms": 100, "end_ms": 200}, {"start_ms": 500, "end_ms": None}]
    valid = interval_is_valid([0, 150, 200, 300, 600], [99, 250, 300, 500, 700], invalid)
    np.testing.assert_equal(valid, [True, False, True, False, False])


def test_connection_change_censors_gap_before_new_snapshot_even_without_bad_delta(tmp_path):
    source(tmp_path, [book(), trade(), book(100, 2, "delta"),
                      trade(300, identifier="b", connection=2), book(5000, 10, connection=2)])
    data, _, _ = read_completed_source(tmp_path)
    assert data["invalid_intervals"] == [{"start_ms": START+100, "end_ms": START+5000,
                                          "reasons": ["connection_changed"]}]
