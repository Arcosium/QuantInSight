"""Synthetic archives only: observed public data provenance stays in the vault."""
import copy
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from quant.impact_backfill import (download_day, load_day, prepare_day,
                                   sources, validate_archive)


DAY = "2026-09-27"
START = 1790467200000


def snapshot(ms=0, update=1, size=10):
    return {"topic": "orderbook.200.ETHUSDT", "type": "snapshot",
            "ts": START + ms + 3, "cts": START + ms,
            "data": {"s": "ETHUSDT", "u": update, "seq": update*10,
                     "b": [[str(99-i), "10"] for i in range(size)],
                     "a": [[str(101+i), "10"] for i in range(size)]}}


def delta(ms, update, bids=(), asks=()):
    row = snapshot(ms, update)
    row["type"] = "delta"
    row["data"]["b"], row["data"]["a"] = list(bids), list(asks)
    return row


def archives(root, books=None, trades=None):
    root.mkdir(parents=True, exist_ok=True)
    source = sources(DAY)
    books = books if books is not None else [snapshot(), delta(100, 2)]
    with zipfile.ZipFile(root / source["orderbook"]["filename"], "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{DAY}_ETHUSDT_ob200.data", "\n".join(json.dumps(row) for row in books) + "\n")
    with gzip.open(root / source["trades"]["filename"], "wt") as stream:
        stream.write("id,timestamp,price,volume,side,rpi\n")
        for row in trades if trades is not None else [("1", START+50, 101, 2, "buy", 0)]:
            stream.write(",".join(map(str, row)) + "\n")
    return root


def prepare(root, **kwargs):
    return prepare_day(root, DAY, keep_depth=5, **kwargs)


def test_retains_deeper_levels_after_best_levels_are_removed(tmp_path):
    root = archives(tmp_path, [snapshot(size=10), delta(100, 2,
        bids=[("99", "0"), ("98", "0"), ("97", "0")],
        asks=[("101", "0"), ("102", "0"), ("103", "0")])])
    data = prepare(root)
    assert isinstance(data["books"], np.memmap)
    assert data["books"].dtype == np.dtype("float64")
    assert data["books"].shape == (2, 2, 5, 2)
    np.testing.assert_equal(data["books"][1, 0, :, 0], [96, 95, 94, 93, 92])
    np.testing.assert_equal(data["books"][1, 1, :, 0], [104, 105, 106, 107, 108])
    assert data["quality"]["max_internal_bid_levels"] == 10
    assert list(data["trades"].columns) == ["ts", "connection", "side", "quantity", "price"]
    assert data["trades"].iloc[0]["connection"] == data["connections"][0]
    np.testing.assert_equal(data["times"], [START, START+100])


def test_same_ms_updates_are_preserved_and_distinct_trade_ids_not_deduplicated(tmp_path):
    root = archives(tmp_path, [snapshot(), delta(100, 2, asks=[("101", "9")]),
                              delta(100, 3, asks=[("101", "7")])],
                    [("1", START+50, 101, 2, "buy", 0),
                     ("2", START+50, 101, 2, "buy", 0),
                     ("2", START+50, 101, 2, "buy", 0),
                     ("3", START+60, 99, 1, "sell", 0),
                     ("4", START+60, 99, 1, "sell", 1)])
    data = prepare(root)
    assert data["quality"]["same_millisecond_book_updates"] == 1
    np.testing.assert_equal(data["books"][:, 1, 0, 1], [10, 9, 7])
    assert list(data["trades"].side) == ["Buy", "Buy", "Sell"]
    assert data["quality"]["duplicate_trade_ids"] == 1
    assert data["quality"]["rpi_trades_excluded"] == 1


def test_conflicting_same_trade_id_is_an_error(tmp_path):
    root = archives(tmp_path, trades=[("1", START+50, 101, 2, "buy", 0),
                                     ("1", START+50, 101, 3, "buy", 0)])
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(root)
    assert not list((root / "cache").glob("*.partial-*"))


def test_snapshots_and_sequence_gaps_never_bridge(tmp_path):
    root = archives(tmp_path, [snapshot(), delta(100, 2), delta(200, 4),
                              delta(300, 5), snapshot(400, 10), delta(500, 11)])
    data = prepare(root)
    np.testing.assert_equal(data["times"], [START, START+100, START+400, START+500])
    assert data["epochs"][0] == data["epochs"][1]
    assert data["epochs"][1] != data["epochs"][2]
    assert data["epochs"][2] == data["epochs"][3]
    assert data["quality"]["update_sequence_gaps"] == 1
    assert data["quality"]["delta_without_valid_snapshot"] == 1


def test_gap_policy_is_explicit_and_in_cache_key(tmp_path):
    root = archives(tmp_path, [snapshot(), delta(2000, 2)])
    strict = prepare(root, max_gap_ms=1000)
    continuous = prepare(root, max_gap_ms=0)
    assert strict["epochs"][0] != strict["epochs"][1]
    assert continuous["epochs"][0] == continuous["epochs"][1]
    assert strict["cache_dir"] != continuous["cache_dir"]
    assert continuous["quality"]["book_gaps_over_1000ms"] == 1
    assert continuous["quality"]["longest_book_gap_ms"] == 2000


@pytest.mark.parametrize("fault,counter", [
    ("missing_cts", "missing_matching_timestamp"),
    ("crossed", "crossed_or_locked_books"),
    ("shallow", "shallow_books"),
    ("reordered", "reordered_or_duplicate_books"),
    ("nan", "malformed_book_messages"),
    ("wrong_symbol", "malformed_book_messages"),
])
def test_invalid_state_requires_fresh_snapshot(tmp_path, fault, counter):
    bad = delta(100, 2)
    if fault == "missing_cts":
        del bad["cts"]  # Publication ts remains; it must not be substituted.
    elif fault == "crossed":
        bad["data"]["b"] = [["102", "3"]]
    elif fault == "shallow":
        bad["data"]["b"] = [[str(99-i), "0"] for i in range(6)]
    elif fault == "reordered":
        bad["data"]["u"] = 1
    elif fault == "nan":
        bad["data"]["a"] = [["101", "nan"]]
    elif fault == "wrong_symbol":
        bad["data"]["s"] = "BTCUSDT"
    root = archives(tmp_path, [snapshot(), bad, delta(200, 3), snapshot(300, 20)])
    data = prepare(root)
    np.testing.assert_equal(data["times"], [START, START+300])
    assert data["quality"][counter] == 1
    assert data["quality"]["delta_without_valid_snapshot"] == 1
    assert data["epochs"][0] != data["epochs"][1]


def test_reordered_snapshot_does_not_make_output_times_unsorted(tmp_path):
    root = archives(tmp_path, [snapshot(100), snapshot(50, 10),
                              delta(200, 11), snapshot(300, 20)])
    data = prepare(root)
    np.testing.assert_equal(data["times"], [START+100, START+300])
    assert data["quality"]["reordered_snapshots"] == 1


def test_outside_day_spill_is_excluded_and_not_silently_bridged(tmp_path):
    root = archives(tmp_path, [snapshot(), delta(100, 2), snapshot(86_400_000, 3)],
                    [("1", START+50, 101, 2, "buy", 0),
                     ("2", START+86_400_000, 101, 2, "buy", 0)])
    data = prepare(root)
    assert len(data["times"]) == 2
    assert len(data["trades"]) == 1
    assert data["quality"]["book_messages_outside_utc_day"] == 1
    assert data["quality"]["trades_outside_utc_day"] == 1


def test_existing_originals_and_provenance_are_not_rewritten(tmp_path):
    root = archives(tmp_path)
    first = download_day(root, DAY)
    tracked = list(root.iterdir())
    mtimes = {path.name: path.stat().st_mtime_ns for path in tracked}
    second = download_day(root, DAY)
    assert first == second
    assert all(path.stat().st_mtime_ns == mtimes[path.name] for path in tracked)
    assert first["files"]["orderbook"]["crc_verified"] is True
    # A valid recompressed but changed original also cannot replace old provenance.
    books = [snapshot(), delta(101, 2)]
    archives(root, books)
    with pytest.raises(ValueError, match="Verified original changed"):
        download_day(root, DAY)


@pytest.mark.parametrize("kind", ["orderbook", "trades"])
def test_archive_crc_or_truncation_errors_are_rejected(tmp_path, kind):
    root = archives(tmp_path)
    path = root / sources(DAY)[kind]["filename"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-8])
    with pytest.raises((ValueError, OSError, EOFError, zipfile.BadZipFile)):
        validate_archive(path, kind)
    original = path.read_bytes()
    with pytest.raises((ValueError, OSError, EOFError, zipfile.BadZipFile)):
        download_day(root, DAY)
    assert path.read_bytes() == original


def test_cache_reuse_and_array_integrity_checks(tmp_path):
    root = archives(tmp_path)
    first = prepare(root)
    manifest_path = Path(first["cache_dir"]) / "manifest.json"
    before = manifest_path.stat().st_mtime_ns
    second = prepare(root)
    assert manifest_path.stat().st_mtime_ns == before
    np.testing.assert_equal(first["books"], second["books"])
    loaded = load_day(first["cache_dir"], mmap=False, verify_hashes=True)
    assert not isinstance(loaded["books"], np.memmap)
    file = Path(first["cache_dir"]) / "times.i64"
    with file.open("r+b") as stream:
        stream.write(b"\0" * 8)
    with pytest.raises(ValueError, match="checksum"):
        load_day(first["cache_dir"], verify_hashes=True)


def test_downloader_has_bounded_retries_and_leaves_no_partial_original(tmp_path, monkeypatch):
    import requests
    from quant import impact_backfill
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(args[0])
        raise requests.ConnectionError("Synthetic unavailable endpoint")
    monkeypatch.setattr(impact_backfill.requests, "get", unavailable)
    monkeypatch.setattr(impact_backfill.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        download_day(tmp_path, DAY, retries=3)
    assert len(attempts) == 3
    assert not list(tmp_path.iterdir())
