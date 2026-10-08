"""Finalized live-capture study with frozen historical cutoffs and local data only.

This module never opens raw captures before their manifest says complete.  It
validates source integrity, excludes invalid book epochs, and reuses the existing
research implementations.  The primary cohort requires quotes <=1 second old;
uncertainty resamples ten-minute blocks, not individual events or invented days.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd

from quant.impact_research import (HORIZONS, reconstruct, first_sustained,
                                   recovery_summary)
from quant.impact_extension import extend
from quant.impact_longitudinal import (GRID, RULES, clean_json, curve_stats,
                                       extract_day, matched_pairs, summarize_cohort,
                                       POINT_INDEX)


BLOCK_MS = 600000
BOOL_COLUMNS = ["valid300", "long_primary", "strict_age_10s", "strict_age_300s",
                "initial_positive", "refill_eligible", "refill_high",
                "absorption_candidate", "spread_widened", "depth_full_coverage",
                "depth_depleted", "opposite_present_10s", "opposite_present_60s",
                "opposite_present_300s"]
NUMERIC_COLUMNS = ["ts", "quantity", "notional", "depth", "spread_ticks", "volatility",
                   "activity", "obi", "directional_obi", "refill_first_seconds",
                   "refill_cycles", "consumed_initial", "half_time_300s",
                   "return_time_300s", "spread_recovery_time", "depth_recovery_time",
                   "impact_5s", "impact_300s", "anchor_price", "anchor_pre_quantity",
                   "refill_ratio_1s", "refill_ratio_10s", "refill_adjusted_10s"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean integer")
    result = int(value)
    if result < 1 or float(result) != float(value):
        raise ValueError("Expected a positive integer")
    return result


def _valid_levels(rows):
    result = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("Malformed price level")
        price, quantity = map(float, row)
        if not np.isfinite([price, quantity]).all() or price <= 0 or quantity < 0:
            raise ValueError("Invalid price level")
        result.append((price, quantity))
    if len(set(p for p, _ in result)) != len(result):
        raise ValueError("Duplicate price in one update")
    return result


def read_completed_source(input_dir):
    """Validate, sanitize, then reuse ``impact_research.reconstruct``.

    Sequence reordering, missing cts or invalid books invalidate the state
    until a fresh snapshot.  Valid observations are never interpolated across
    that boundary.  Source raw files remain untouched.  CRC, manifest counts
    and before/after hashes protect against truncated or still-changing input.
    """
    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("complete") is not True:
        raise ValueError("Capture is not complete; raw data was not opened")
    if manifest.get("depth") != 50 or not manifest.get("symbol"):
        raise ValueError("A completed 50-level capture manifest is required")
    planned = float(manifest["planned_seconds"])
    elapsed = float(manifest["elapsed_seconds"])
    begin = datetime.fromisoformat(manifest["started_utc"])
    end = datetime.fromisoformat(manifest["ended_utc"])
    if planned <= 0 or elapsed < planned-.5 or (end-begin).total_seconds() < planned-.5:
        raise ValueError("Completed flag disagrees with planned capture duration")
    if begin.tzinfo is None or end.tzinfo is None:
        raise ValueError("Capture manifest timestamps must include timezone")
    raw_path = input_dir / "raw.jsonl.gz"
    before_stat = raw_path.stat()
    raw_hash = file_sha256(raw_path)
    expected_hash = manifest.get("raw_sha256")
    if expected_hash is not None and expected_hash != raw_hash:
        raise ValueError("Source raw SHA256 differs from manifest")
    quality, counts = Counter(), Counter()
    active_conn, last_u, last_ts, emitted_ts = None, None, None, None
    last_seq, last_payload = None, None
    bids, asks, seen_trades = {}, {}, {}
    book_digest = hashlib.sha256()
    symbol = manifest["symbol"]
    invalid_intervals, invalid_reasons = [], set()
    invalid_since = None
    last_record_connection = None
    def invalidate(reason):
        nonlocal active_conn, invalid_since
        active_conn = None
        if invalid_since is None:
            # A malformed or reordered message may not provide a usable cts.
            # Conservatively censor from the last verified matching-engine time.
            invalid_since = emitted_ts if emitted_ts is not None else int(begin.timestamp()*1000)
        invalid_reasons.add(reason)
    def close_invalid_interval(snapshot_time):
        nonlocal invalid_since
        if invalid_since is not None:
            invalid_intervals.append({"start_ms": invalid_since, "end_ms": snapshot_time,
                                      "reasons": sorted(invalid_reasons)})
            invalid_since = None
            invalid_reasons.clear()
    # This temporary stream contains only public messages from this source.
    with tempfile.TemporaryDirectory(prefix="impact-live-validated-", dir=input_dir.parent) as temp:
        validated = Path(temp) / "validated.jsonl.gz"
        with gzip.open(raw_path, "rt", encoding="utf-8") as source, gzip.open(validated, "wt", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                    msg, conn = row["message"], _positive_integer(row["segment"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Malformed source record at line {line_number}") from error
                topic = msg.get("topic", msg.get("op", "other"))
                counts[topic] += 1
                if last_record_connection is not None and conn != last_record_connection:
                    if conn < last_record_connection:
                        raise ValueError("Capture connection segments are reordered")
                    invalidate("connection_changed")
                    quality["connection_changes"] += 1
                last_record_connection = conn
                if topic.startswith("orderbook."):
                    quality["book_messages"] += 1
                    try:
                        item = msg["data"]
                        if topic != f"orderbook.50.{symbol}" or item["s"] != symbol:
                            raise ValueError("Unexpected orderbook symbol/depth")
                        kind = msg["type"]
                        if kind not in {"snapshot", "delta"}:
                            raise ValueError("Unexpected book message type")
                        if msg.get("cts") is None:
                            quality["missing_matching_timestamp"] += 1
                            invalidate("missing_matching_timestamp")
                            continue
                        ts, update = _positive_integer(msg["cts"]), _positive_integer(item["u"])
                        seq = _positive_integer(item["seq"]) if "seq" in item else None
                        changes = (_valid_levels(item["b"]), _valid_levels(item["a"]))
                    except (KeyError, TypeError, ValueError, OverflowError):
                        quality["malformed_book_messages"] += 1
                        invalidate("malformed_book_message")
                        continue
                    if kind == "snapshot":
                        if emitted_ts is not None and ts < emitted_ts:
                            quality["reordered_snapshots"] += 1
                            invalidate("reordered_snapshot")
                            continue
                        quality["snapshots"] += 1
                        bids, asks = {}, {}
                        active_conn, last_u, last_ts = conn, None, None
                        last_seq, last_payload = None, None
                    elif conn != active_conn:
                        quality["delta_without_valid_snapshot"] += 1
                        invalidate("delta_without_valid_snapshot")
                        continue
                    if last_u is not None:
                        if update == last_u and ts == last_ts and item == last_payload:
                            quality["identical_duplicate_book_updates"] += 1
                            continue
                        if update <= last_u or ts < last_ts or (seq is not None and last_seq is not None and seq < last_seq):
                            quality["reordered_or_conflicting_duplicate_book"] += 1
                            invalidate("reordered_or_conflicting_duplicate_book")
                            continue
                        if update > last_u+1:
                            # Official docs identify u as an Update ID but do not
                            # guarantee increments of exactly one for this feed.
                            quality["nonunit_update_id_increments"] += 1
                        if ts-last_ts > 1000:
                            quality["book_update_gaps_over_1000ms"] += 1
                        if ts == last_ts:
                            quality["same_millisecond_book_updates"] += 1
                    for state, updates in ((bids, changes[0]), (asks, changes[1])):
                        for price, quantity in updates:
                            if quantity == 0:
                                state.pop(price, None)
                            else:
                                state[price] = quantity
                    b, a = sorted(bids.items(), reverse=True)[:50], sorted(asks.items())[:50]
                    if len(b) < 50 or len(a) < 50 or b[0][0] >= a[0][0]:
                        quality["invalid_crossed_or_shallow_books"] += 1
                        invalidate("invalid_crossed_or_shallow_book")
                        continue
                    bids, asks = dict(b), dict(a)
                    if kind == "snapshot":
                        close_invalid_interval(ts)
                    last_u, last_ts, emitted_ts = update, ts, ts
                    last_seq, last_payload = seq, item
                    book_digest.update(np.asarray([b, a], dtype=np.float64).tobytes())
                    quality["valid_books"] += 1
                    target.write(json.dumps(row, separators=(",", ":"))+"\n")
                elif topic.startswith("publicTrade."):
                    if topic != f"publicTrade.{symbol}":
                        raise ValueError("Unexpected public trade symbol")
                    accepted = []
                    for trade in msg["data"]:
                        quality["trade_rows"] += 1
                        try:
                            tid = str(trade["i"])
                            ts = _positive_integer(trade["T"])
                            price, quantity = float(trade["p"]), float(trade["v"])
                            if (not tid or trade["S"] not in {"Buy", "Sell"}
                                    or trade.get("s", symbol) != symbol
                                    or not np.isfinite([price, quantity]).all()
                                    or price <= 0 or quantity <= 0):
                                raise ValueError("Invalid public trade")
                        except (KeyError, TypeError, ValueError, OverflowError):
                            quality["invalid_trades"] += 1
                            continue
                        identity = (ts, trade["S"], price, quantity, bool(trade.get("BT")), bool(trade.get("RPI")))
                        if tid in seen_trades:
                            if seen_trades[tid] != identity:
                                raise ValueError("Conflicting duplicate public trade ID")
                            quality["duplicate_trade_ids"] += 1
                            continue
                        seen_trades[tid] = identity
                        if trade.get("BT") or trade.get("RPI"):
                            quality["block_or_rpi_trades_excluded"] += 1
                            continue
                        accepted.append(trade)
                    if accepted:
                        target.write(json.dumps({**row, "message": {**msg, "data": accepted}}, separators=(",", ":"))+"\n")
        # Reading to EOF above validates gzip trailers, including concatenated members.
        if dict(counts) != manifest.get("counts"):
            raise ValueError("Raw message counts differ from completed manifest")
        after_stat = raw_path.stat()
        if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
            raise ValueError("Source changed during validation")
        if file_sha256(raw_path) != raw_hash or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Source or manifest changed during validation")
        data = reconstruct(validated)
    if invalid_since is not None:
        invalid_intervals.append({"start_ms": invalid_since, "end_ms": None,
                                  "reasons": sorted(invalid_reasons)})
    if len(data["books"]) != quality["valid_books"] or hashlib.sha256(data["books"].tobytes()).hexdigest() != book_digest.hexdigest():
        raise ValueError("Legacy reconstruction disagrees with validated book stream")
    quality["valid_trades"] = len(data["trades"])
    quality["known_invalid_intervals"] = len(invalid_intervals)
    data["quality"] = dict(quality)
    data["invalid_intervals"] = invalid_intervals
    data["trades"] = data["trades"].sort_values("ts", kind="stable").reset_index(drop=True)
    source = {"input_dir": str(input_dir), "raw_sha256": raw_hash,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "gzip_crc_verified": True, "manifest_complete": True,
              "message_counts_verified": True, "raw_bytes": before_stat.st_size,
              "source": manifest["source"], "depth": 50, "symbol": symbol,
              "planned_seconds": planned, "elapsed_seconds": elapsed,
              "started_utc": manifest["started_utc"], "ended_utc": manifest["ended_utc"],
              "collector_errors": manifest.get("errors", []),
              "known_invalid_intervals": invalid_intervals}
    return data, manifest, source


def frozen_regimes(frame, reference):
    """Classify using reference cutoffs only; never fit on this capture."""
    frame = frame.copy()
    for field in ["depth", "volatility", "activity"]:
        lo, hi = map(float, reference["regime_thresholds"][field]["limits"])
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("Invalid frozen regime limits")
        values = pd.to_numeric(frame[field], errors="coerce")
        frame[field+"_regime"] = np.where(values.isna(), "Unclassified",
            np.where(values <= lo, "Low", np.where(values >= hi, "High", "Normal")))
    frame["spread_regime"] = np.where(frame.spread_ticks == 1, "1 tick",
                                     np.where(frame.spread_ticks == 2, "2 tick", "3+ tick"))
    frame["obi_bin"] = pd.cut(frame.directional_obi, [-1.000001, -.2, .2, 1.000001],
                               right=False, labels=["negative", "neutral", "positive"]).astype(str)
    return frame


def _complete_columns(frame):
    frame = frame.copy()
    for field in NUMERIC_COLUMNS:
        if field not in frame:
            frame[field] = np.nan
    for field in BOOL_COLUMNS:
        if field not in frame:
            frame[field] = False
        frame[field] = frame[field].eq(True)
    for field in ["day", "side", "refill_exclusion"]:
        if field not in frame:
            frame[field] = ""
    return frame


def rename_blocks(value):
    if isinstance(value, dict):
        names = {"days": "time_blocks", "cohort_days": "cohort_time_blocks"}
        return {names.get(key, key): rename_blocks(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename_blocks(item) for item in value]
    return value


def block_stats(values, blocks):
    result = rename_blocks(curve_stats(values, blocks))
    result["clusters"] = result["time_blocks"]
    return result


def interval_is_valid(starts, ends, invalid_intervals):
    """A next snapshot ends censoring; never carry an old epoch into a bad span."""
    starts, ends = np.broadcast_arrays(np.asarray(starts), np.asarray(ends))
    valid = np.ones(starts.shape, dtype=bool)
    for interval in invalid_intervals:
        bad_end = interval["end_ms"]
        valid &= ~((ends >= interval["start_ms"])
                   & ((starts < bad_end) if bad_end is not None else True))
    return valid


def calibration_from_events(events):
    result = {}
    for name in ["all", "thin", "thick"]:
        selected = [event for event in events if event["depleted"]
                    and (name == "all" or event["regime"] == name)]
        stat = recovery_summary([event["t50_gap"] for event in selected], len(selected))
        median = stat["median_seconds"]
        empirical = stat["n"] >= 8 and median is not None and median > .1
        result[name] = {"half_time_seconds": round(median-.1, 3) if empirical else None,
                        "depletion_events": stat["n"], "recovered": stat["recovered"],
                        "censored": stat["censored"], "empirical": empirical,
                        "source": "current live fixed-band net-depth deficit half recovery, all-event median" if empirical else "current live observations insufficient or censored",
                        "recovery_median_from_event_start_seconds": median,
                        "initial_depth_time_seconds": .1}
    return result


def select_snapshots(data, frame):
    snapshots = []
    for regime, label in [("thin", "Low"), ("thick", "High")]:
        for side in ["buy", "sell"]:
            selected = frame[(frame.depth_regime == label) & (frame.side == side)]
            if selected.empty:
                continue
            median = selected.depth.median()
            # Tie-break by event time, independent of any subsequent response.
            selected = selected.assign(distance=(selected.depth-median).abs()).sort_values(["distance", "ts"])
            event = selected.iloc[0]
            pre = int(np.searchsorted(data["times"], int(event.ts), side="left")-1)
            snapshots.append({"id": f"{regime}-{side}", "regime": regime, "event_side": side,
                              "timestamp_ms": int(data["times"][pre]), "event_timestamp_ms": int(event.ts),
                              "pre_depth5": float(event.depth), "group_median_depth5": float(median),
                              "selection": "closest pre-event depth to current live group median; earliest-time tie-break",
                              "bids": data["books"][pre, 0].tolist(), "asks": data["books"][pre, 1].tolist()})
    return snapshots


def make_legacy_events(data, frame):
    times, books = data["times"], data["books"]
    mid = books[:, :, 0, 0].mean(axis=1)
    result, excluded = [], Counter()
    for row in frame.itertuples():
        t = int(row.ts)
        pre = int(np.searchsorted(times, t, side="left")-1)
        before = int(np.searchsorted(times, t-1000, side="right")-1)
        future = np.searchsorted(times, t+(HORIZONS*1000).astype(int), side="right")-1
        side, sign = (1, 1) if row.side == "buy" else (0, -1)
        anchor = books[pre, side, :5]
        low, high = anchor[:, 0].min(), anchor[:, 0].max()
        prices, quantities = books[future, side, :, 0], books[future, side, :, 1]
        coverage = prices[:, -1] >= high if side == 1 else prices[:, -1] <= low
        if not coverage.all():
            excluded["fixed_pre5_band_outside_retained_depth"] += 1
            continue
        depth = float(anchor[:, 1].sum())
        fixed = np.where((prices >= low) & (prices <= high), quantities, 0).sum(axis=1)/depth
        rolling = books[future, side, :5, 1].sum(axis=1)/depth
        spread = (books[future, 1, 0, 0]-books[future, 0, 0, 0])/mid[pre]*1e4
        pre_spread = (books[pre, 1, 0, 0]-books[pre, 0, 0, 0])/mid[pre]*1e4
        depleted, widened = bool(fixed[0] <= .8), bool(spread[0] > pre_spread+1e-8)
        result.append({"ts": t, "minute": t//BLOCK_MS, "side": row.side,
                       "quantity": float(row.quantity), "notional": float(row.notional),
                       "depth": depth, "pre_spread_bps": float(pre_spread),
                       "pre_return_bps": float(sign*(mid[pre]-mid[before])/mid[before]*1e4),
                       "impact": sign*(mid[future]-mid[pre])/mid[pre]*1e4,
                       "depth_fixed": fixed, "depth_rolling": rolling, "spread": spread,
                       "depleted": depleted, "t90": first_sustained(fixed, .9, after_index=1) if depleted else None,
                       "t50_gap": first_sustained(fixed, (1+fixed[0])/2, after_index=1) if depleted else None,
                       "spread_widened": widened,
                       "spread_recovery_seconds": first_sustained(spread, pre_spread+1e-8, above=False, after_index=1) if widened else None,
                       "book_index": pre, "regime": {"Low": "thin", "Normal": "middle", "High": "thick"}[row.depth_regime]})
    return result, dict(excluded)


def _legacy_pairing(events, field, low, high):
    if not events:
        return {"pairs": 0, "difference_low_minus_high": block_stats([], []), "indices": []}
    frame = pd.DataFrame(events)
    frame["day"] = frame.ts//BLOCK_MS
    if "obi_bin" not in frame:
        frame["obi_bin"] = "unused"
    pairs = matched_pairs(frame, field, low, high)
    return {"pairs": len(pairs), "quantity_ratio_limit": 1.1,
            "difference_low_minus_high": block_stats([events[a]["impact"]-events[b]["impact"] for a, b in pairs],
                                                       [events[a]["minute"] for a, b in pairs]),
            "interpretation": "same side, quantity ratio <=1.1, same ten-minute block; descriptive block-bootstrap interval",
            "indices": pairs}


def build_envelope(data, manifest, reference, frame, live):
    primary = frame[frame.primary]
    events, exclusions = make_legacy_events(data, primary)
    def stats(selected, key):
        return block_stats([event[key] for event in selected], [event["minute"] for event in selected])
    groups = {}
    for name in ["all", "thin", "middle", "thick"]:
        selected = events if name == "all" else [event for event in events if event["regime"] == name]
        depleted = [event for event in selected if event["depleted"]]
        widened = [event for event in selected if event["spread_widened"]]
        groups[name] = {"n": len(selected), "median_quantity": float(np.median([e["quantity"] for e in selected])) if selected else None,
                        "median_depth": float(np.median([e["depth"] for e in selected])) if selected else None,
                        "impact": stats(selected, "impact"), "spread": stats(selected, "spread"),
                        "pre_spread_mean_bps": float(np.mean([e["pre_spread_bps"] for e in selected])) if selected else None,
                        "spread_recovery": recovery_summary([e["spread_recovery_seconds"] for e in widened], len(widened)),
                        "depth_fixed": stats(depleted, "depth_fixed"), "depth_rolling": stats(depleted, "depth_rolling"),
                        "t90": recovery_summary([e["t90"] for e in depleted], len(depleted)),
                        "t50_gap": recovery_summary([e["t50_gap"] for e in depleted], len(depleted))}
    paired = _legacy_pairing(events, "regime", "thin", "thick")
    pairs = paired.pop("indices")
    matched = {"pairs": len(pairs), "difference": paired["difference_low_minus_high"],
               "thin": stats([events[a] for a, _ in pairs], "impact"),
               "thick": stats([events[b] for _, b in pairs], "impact"),
               "quantity_ratio_median": float(np.median([max(events[a]["quantity"], events[b]["quantity"])/min(events[a]["quantity"], events[b]["quantity"]) for a, b in pairs])) if pairs else None,
               "inference": paired.get("interpretation", "No eligible matched pairs")}
    average = np.asarray(groups["all"]["impact"]["mean"])
    peak = int(np.argmax(average)) if len(average) else None
    half = first_sustained(average, average[peak]/2, above=False, after_index=peak+1) if peak is not None and average[peak] > 0 else None
    decay = {"peak_bps": float(average[peak]) if peak is not None else None,
             "peak_at_seconds": float(HORIZONS[peak]) if peak is not None else None,
             "half_level_at_seconds": half,
             "half_decay_seconds": half-float(HORIZONS[peak]) if half is not None else None,
             "end_bps": float(average[-1]) if len(average) else None}
    times = data["times"]
    base = {"schema_version": 1, "symbol": manifest["symbol"], "venue": "Bybit spot",
            "started_utc": datetime.fromtimestamp(times[0]/1000, timezone.utc).isoformat(),
            "ended_utc": datetime.fromtimestamp(times[-1]/1000, timezone.utc).isoformat(),
            "duration_seconds": float((times[-1]-times[0])/1000),
            "book_updates": len(times), "trades": len(data["trades"]),
            "large_notional_threshold": reference["notional_threshold"],
            "quality": data["quality"], "exclusions": exclusions,
            "method": {**live["method"], "followup_seconds": 10},
            "regime_depth_thresholds": reference["regime_thresholds"]["depth"]["limits"],
            "horizons_seconds": HORIZONS.tolist(), "groups": groups, "matched": matched,
            "decay": decay, "snapshots": select_snapshots(data, primary),
            "calibration": calibration_from_events(events), "live_study": live,
            "limitations": live["limitations"] + ["호환 화면의 고정 가격 구간 통계는 10초간 해당 가격 구간이 50단계 안에서 관측된 사건만 사용한다."],
            "compatibility_event_count": len(events), "primary_event_count": int(frame.primary.sum())}
    # extend() needs two endpoints even when there are no events.  This temporary
    # shape protects its empty-cohort path; it does not introduce observations.
    extension_base = base if events else {**base, "groups": {**groups, "all": {**groups["all"], "impact": {"mean": [None]*len(HORIZONS)}}}}
    extension, enriched, _, market = extend(data, extension_base, events)
    # The legacy extension fits local quantiles; replace that entire presentation
    # with frozen reference classifications and same-block matching.
    if enriched:
        frozen = frozen_regimes(pd.DataFrame(enriched), reference)
        for i, record in enumerate(enriched):
            for field in ["depth_regime", "volatility_regime", "activity_regime", "obi_bin"]:
                record[field] = frozen.iloc[i][field]
    regimes = {}
    for field in ["depth", "volatility", "activity", "spread"]:
        labels = ["1 tick", "2 tick", "3+ tick"] if field == "spread" else ["Low", "Normal", "High", "Unclassified"]
        pairing = _legacy_pairing(enriched, field+"_regime", labels[0], labels[2])
        pairing.pop("indices")
        regimes[field] = {"groups": {label: stats([r for r in enriched if r[field+"_regime"] == label], "impact") for label in labels}, "matched": pairing}
        if field != "spread":
            regimes[field].update(thresholds=reference["regime_thresholds"][field]["limits"],
                                  quantiles=reference["regime_thresholds"][field]["quantiles"], frozen=True)
    extension["regimes"] = regimes
    extension["obi_cells"] = [{"lower": lower, "upper": min(upper, 1), "depth": label,
                               "impact": stats([r for r in enriched if lower <= r["directional_obi"] < upper and r["depth_regime"] == label], "impact")}
                              for lower, upper in [(-1, -.2), (-.2, .2), (.2, 1.000001)] for label in ["Low", "High"]]
    extension["method"].update(regime_threshold_source="frozen first 60 historical days",
                                uncertainty_unit="ten-minute block", matching_same_block=True,
                                live_duration_seconds=base["duration_seconds"])
    extension["limitations"] = [text for text in extension["limitations"] if "30분" not in text] + live["limitations"]
    extension["started_utc"], extension["ended_utc"] = base["started_utc"], base["ended_utc"]
    base["extension"], base["pilot_market"] = extension, market
    return clean_json(base), events


def analyse_live(data, manifest, reference):
    times = data["times"]
    dates = pd.to_datetime([times[0], times[-1]], unit="ms", utc=True).strftime("%Y-%m-%d").tolist()
    if dates[0] != dates[1]:
        raise ValueError("This short live study must be contained in one UTC date")
    threshold = float(reference["notional_threshold"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid frozen notional cutoff")
    frame, curves, post, metadata = extract_day(data, dates[0], threshold)
    frame = _complete_columns(frame).reset_index(drop=True)
    if frame.empty:
        curves = post = np.empty((0, len(GRID)))
    invalid_intervals = data.get("invalid_intervals", [])
    starts = frame.ts.to_numpy(np.int64)-10000
    frame["known_valid_10s"] = interval_is_valid(starts, frame.ts.to_numpy(np.int64)+10000, invalid_intervals)
    frame["known_valid_300s"] = interval_is_valid(starts, frame.ts.to_numpy(np.int64)+300000, invalid_intervals)
    # Null each contaminated response horizon, even for the age-relaxed
    # sensitivity.  Neither elapsed time nor a later snapshot repairs that gap.
    if len(frame):
        targets = frame.ts.to_numpy(np.int64)[:, None]+np.rint(GRID*1000).astype(np.int64)
        curves[~interval_is_valid(starts[:, None], targets, invalid_intervals)] = np.nan
        end_targets = frame.event_end_ms.to_numpy(np.int64)[:, None]+np.rint(GRID*1000).astype(np.int64)
        post[~interval_is_valid(starts[:, None], end_targets, invalid_intervals)] = np.nan
        for horizon, index in POINT_INDEX.items():
            frame[f"impact_{horizon:g}s"] = curves[:, index]
    invalid_long_candidates = int((frame.long_primary & ~frame.known_valid_300s).sum())
    frame["long_primary"] &= frame.known_valid_300s
    frame["valid300"] &= frame.known_valid_300s
    frame["strict_age_300s"] &= frame.known_valid_300s
    frame.loc[~frame.known_valid_300s, ["half_time_300s", "return_time_300s",
                                      "spread_recovery_time", "depth_recovery_time"]] = np.nan
    frame.loc[~frame.known_valid_10s, "refill_eligible"] = False
    frame.loc[~frame.known_valid_10s, "refill_exclusion"] = "known_invalid_book_interval"
    frame = frozen_regimes(frame, reference)
    frame["time_block_10m"] = (frame.ts//BLOCK_MS).astype("int64")
    frame["primary"] = frame.strict_age_10s & frame.known_valid_10s
    # Refuse mixed-connection trade buckets even when book timestamps align.
    connection_ok = []
    trades = data["trades"]
    for row in frame.itertuples():
        pre = int(np.searchsorted(times, row.ts, side="left")-1)
        initial = trades[(trades.ts >= row.ts) & (trades.ts < row.ts+100)]
        connection_ok.append(bool(len(initial) and (initial.connection == data["connections"][pre]).all()))
    frame["connection_consistent"] = np.asarray(connection_ok, dtype=bool)
    frame["primary"] &= frame.connection_consistent
    stats_frame = frame.copy()
    stats_frame["day"] = frame.time_block_10m
    start_ms = int(times[0])
    masks = {"all": frame.primary,
             "first_hour": frame.primary & (frame.ts < start_ms+3600000),
             "second_hour": frame.primary & (frame.ts >= start_ms+3600000),
             "strict_age_300s": frame.primary & frame.strict_age_300s,
             "all_contiguous_sensitivity": frame.connection_consistent & frame.known_valid_10s}
    with warnings.catch_warnings():
        # The post-classification curve deliberately has no values before 1 s,
        # and boundary-censored horizons can be entirely unobserved.  Their
        # JSON values remain null; NumPy's expected all-NaN warning adds no data.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        cohorts = {name: rename_blocks(summarize_cohort(stats_frame, curves, post, mask)) for name, mask in masks.items()}
    method = {**RULES, "clock": "live book cts / public trade T, milliseconds",
              "notional_threshold_policy": "frozen first 60 historical days; no fit on live capture",
              "regime_threshold_policy": "all three regime cutoffs frozen to historical reference",
              "primary_quote_age_limit_ms": 1000, "time_block_ms": BLOCK_MS,
              "matching_same_time_block": True,
              "inference": "ten-minute block bootstrap; pointwise exploratory 95% intervals; one block has no interval",
              "time_split": "first/second 60 minutes are descriptive subperiods, not training or validation",
              "long_cohort_policy": "301-second spacing and contiguous epoch; <=1-second age is reported separately",
              "long_selection_order": "select 301-second-spaced events before the strict 10-second age filter; the primary long cohort is a conservative subset, not reselected",
              "known_invalid_policy": "censor from last valid cts before a detected invalid book through next valid snapshot; no horizon or lookback may cross, including age-relaxed sensitivity",
              "update_id_policy": "monotonic u/seq and cts; non-unit positive u increments counted, not assumed to prove packet loss"}
    method.pop("archive_update_policy", None)
    live = {"schema_version": 1, "symbol": manifest["symbol"], "venue": "Bybit spot",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "events": int(frame.primary.sum()), "extracted_events": len(frame),
            "excluded_from_primary": {"quote_age_over_1000ms": int((~frame.strict_age_10s).sum()),
                                      "connection_mismatch": int((~frame.connection_consistent).sum()),
                                      "known_invalid_interval": int((~frame.known_valid_10s).sum())},
            "known_invalid_intervals": invalid_intervals,
            "known_invalid_long_windows": int((~frame.known_valid_300s).sum()),
            "known_invalid_long_candidates": invalid_long_candidates,
            "book_updates": len(times), "trades": len(trades), "dates": [dates[0]],
            "notional_threshold": threshold, "regime_thresholds": reference["regime_thresholds"],
            "reference_calibration_end": reference.get("calibration_end"),
            "horizons_seconds": GRID.tolist(), "method": method, "cohorts": cohorts,
            "quality": data["quality"], "extraction": metadata, "actual_own_orders_observed": 0,
            "started_utc": datetime.fromtimestamp(times[0]/1000, timezone.utc).isoformat(),
            "ended_utc": datetime.fromtimestamp(times[-1]/1000, timezone.utc).isoformat(),
            "duration_seconds": float((times[-1]-times[0])/1000),
            "limitations": ["이번에 완료한 한 종목의 실시간 수집 구간만 관측 표본으로 사용했다. 과거 자료는 사전에 고정한 분류 문턱에만 사용했다.",
                "체결 묶음은 익명 시장 흐름이며 특정 투자자의 한 주문을 식별한 것이 아니다.",
                "표시 잔량의 양의 변화는 신규 공급량의 대용치이며 취소·체결·숨은 유동성을 완전히 분리하지 못한다.",
                "300초 가격 변화에는 후속 주문과 시장 공통 움직임이 포함되므로 최초 체결의 인과효과나 영구 충격으로 해석하지 않는다.",
                "신뢰구간은 10분 블록을 재표집한 점별 탐색 구간이다. 한 블록만 있으면 구간을 추정하지 않으며 다중 비교를 보정하지 않았다.",
                "첫 60분과 이후 구간은 시간대별 기술통계이며 훈련·검증 분리가 아니다.",
                "증가한 update ID의 간격만으로 패킷 누락을 단정하지 않는다. 비단위 증가를 따로 기록하며 시각·순서 역행과 호가 오류는 새로운 snapshot까지 제외한다.",
                "실제 본인 주문 체결 관측은 0건이다. ArcTrade의 기존 가상 주문 모형 실행과 실제 시장 관측을 구분한다."]}
    live = clean_json(live)
    envelope, legacy_events = build_envelope(data, manifest, reference, frame, live)
    return envelope, frame, curves, post, legacy_events


def run(input_dir, reference_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output must be a new directory; previous results are preserved")
    reference_path = Path(reference_path)
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    data, manifest, source = read_completed_source(input_dir)
    result, frame, curves, post, legacy_events = analyse_live(data, manifest, reference)
    if reference_path.read_bytes() != reference_bytes:
        raise ValueError("Frozen reference changed during analysis")
    provenance = {"source": source, "reference_path": str(reference_path),
                  "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                  "source_code_sha256": file_sha256(__file__),
                  "started_utc": result["started_utc"], "ended_utc": result["ended_utc"],
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "complete": True, "study": "full", "actual_own_orders_observed": 0}
    result["live_study"]["provenance"] = provenance
    result["provenance"] = provenance
    # Serialization is checked before publishing any final output.
    study_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
    live_json = json.dumps(result["live_study"], ensure_ascii=False, allow_nan=False)
    output.mkdir(parents=True)
    frame.to_csv(output / "events.csv", index=False)
    calibration_rows = []
    for event in legacy_events:
        row = {key: event[key] for key in ["ts", "book_index", "side", "regime", "depth", "depleted", "t50_gap", "t90"]}
        row["depth_ratio_at_0_1s"] = float(event["depth_fixed"][0])
        row["half_deficit_target_ratio"] = float((1+event["depth_fixed"][0])/2)
        row["depth_fixed_ratios_json"] = json.dumps(event["depth_fixed"].tolist(), allow_nan=False)
        calibration_rows.append(row)
    pd.DataFrame(calibration_rows, columns=["ts", "book_index", "side", "regime", "depth", "depleted", "t50_gap", "t90", "depth_ratio_at_0_1s", "half_deficit_target_ratio", "depth_fixed_ratios_json"]).to_csv(output / "calibration_events.csv", index=False)
    np.savez_compressed(output / "curves.npz", impact=curves, after_end=post)
    (output / "live_study.json").write_text(live_json, encoding="utf-8")
    (output / "study.json").write_text(study_json, encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=["full"], default="full")
    args = parser.parse_args()
    result = run(args.input, args.reference, args.output)
    print(json.dumps({"output": str(args.output), "events": result["live_study"]["events"],
                      "long_events": result["live_study"]["cohorts"]["all"]["resiliency"]["cohort_n"],
                      "snapshots": [s["id"] for s in result["snapshots"]],
                      "calibration": result["calibration"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
