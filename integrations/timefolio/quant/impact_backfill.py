"""Bybit public spot archive ingestion without modifying source archives.

The source book has 200 levels.  All reported levels are reconstructed before
only ``keep_depth`` levels are written to a bounded-memory, per-day cache.
Matching-engine ``cts`` is mandatory; publication ``ts`` is never substituted.
Each UTC day and each snapshot starts a new epoch.  Gaps over ``max_gap_ms``
also start a new epoch unless that option is zero.  Regardless of this option,
gaps over one second are reported. Sequence loss invalidates the state until a
snapshot; a continuous sequence with no updates can reflect an unchanged book.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import requests

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads


CACHE_VERSION = 1
TRADE_COLUMNS = ["ts", "connection", "side", "quantity", "price"]


def _identifiers(day, symbol):
    day = date.fromisoformat(str(day)).isoformat()
    symbol = str(symbol).upper()
    if not re.fullmatch(r"[A-Z0-9]{3,30}", symbol):
        raise ValueError("Invalid exchange symbol")
    return day, symbol


def sources(day, symbol="ETHUSDT"):
    day, symbol = _identifiers(day, symbol)
    book = f"{day}_{symbol}_ob200.data.zip"
    trades = f"{symbol}_{day}.csv.gz"
    return {
        "orderbook": {"filename": book,
                      "url": f"https://quote-saver.bycsi.com/orderbook/spot/{symbol}/{book}"},
        "trades": {"filename": trades,
                   "url": f"https://public.bybit.com/spot/{symbol}/{trades}"},
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path, kind):
    """Read the entire compressed stream, including ZIP/GZIP CRC trailers."""
    path = Path(path)
    if kind == "orderbook":
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) != 1 or not entries[0].filename.endswith(".data"):
                raise ValueError("Expected exactly one .data member in orderbook ZIP")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP CRC validation failed: {bad}")
            return {"member": entries[0].filename,
                    "uncompressed_bytes": entries[0].file_size, "crc_verified": True}
    if kind == "trades":
        total = 0
        with gzip.open(path, "rb") as stream:
            first = stream.readline()
            if first.decode("utf-8-sig").strip() != "id,timestamp,price,volume,side,rpi":
                raise ValueError("Unexpected public spot trade CSV header")
            total += len(first)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
        return {"uncompressed_bytes": total, "crc_verified": True}
    raise ValueError(f"Unknown archive kind: {kind}")


def _write_json_exclusive(path, value):
    # Use an exclusive create: never replace provenance of an earlier run.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _download(url, destination, kind, retries):
    if retries < 1 or retries > 5:
        raise ValueError("retries must be between 1 and 5")
    last_error = None
    for attempt in range(retries):
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                         dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            with requests.get(url, stream=True, timeout=(15, 120),
                              headers={"Accept-Encoding": "identity"}) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                with temporary.open("wb") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            target.write(chunk)
                if expected and temporary.stat().st_size != int(expected):
                    raise ValueError("Downloaded byte count differs from Content-Length")
            validate_archive(temporary, kind)
            # Hard link makes publication atomic and refuses to replace a file
            # another worker might have created while this download was active.
            os.link(temporary, destination)
            return
        except FileExistsError:
            validate_archive(destination, kind)
            return
        except (requests.RequestException, OSError, ValueError, EOFError,
                zipfile.BadZipFile) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Archive download failed after {retries} attempts: {url}") from last_error


def download_day(root, day, symbol="ETHUSDT", retries=3):
    """Download missing archives or verify existing originals; return provenance.

    Existing files are never overwritten, even if corrupt.  Per-day provenance
    is kept in ``backfill_sources_<symbol>_<day>.json``; unrelated downloader
    manifests are not opened or modified.
    """
    day, symbol = _identifiers(day, symbol)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / f"backfill_sources_{symbol}_{day}.json"
    old = json.loads(provenance.read_text()) if provenance.exists() else None
    files = {}
    for kind, source in sources(day, symbol).items():
        path = root / source["filename"]
        existed = path.exists()
        if not existed:
            _download(source["url"], path, kind, retries)
        verification = validate_archive(path, kind)
        digest = sha256(path)
        if old and old["files"][kind]["sha256"] != digest:
            raise ValueError(f"Verified original changed; refusing to replace it: {path}")
        files[kind] = dict(source, **verification, sha256=digest,
                           bytes=path.stat().st_size, adopted_existing=existed)
    if old:
        return old
    manifest = {"schema_version": 1, "venue": "Bybit", "market": "spot",
                "symbol": symbol, "day_utc": day,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(), "files": files}
    _write_json_exclusive(provenance, manifest)
    return manifest


def _day_bounds(day):
    start = datetime.combine(date.fromisoformat(day), datetime.min.time(), timezone.utc)
    return int(start.timestamp() * 1000), int((start + timedelta(days=1)).timestamp() * 1000)


def _integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a timestamp or sequence")
    result = int(value)
    if isinstance(value, float) and result != value:
        raise ValueError("Fractional timestamp or sequence")
    if isinstance(value, str) and str(result) != value.strip():
        raise ValueError("Noncanonical integer")
    return result


def _levels(rows):
    result = []
    for row in rows:
        if len(row) != 2:
            raise ValueError("A level must contain price and quantity")
        price, quantity = float(row[0]), float(row[1])
        if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity < 0:
            raise ValueError("Invalid price or quantity")
        result.append((price, quantity))
    if len({price for price, _ in result}) != len(result):
        raise ValueError("A book message contains duplicate prices")
    return result


class _BookWriter:
    def __init__(self, directory, keep_depth, chunk_size=2048):
        self.directory = directory
        self.handles = {key: (directory / name).open("xb") for key, name in
                        {"books": "books.f64", "times": "times.i64",
                         "epochs": "epochs.i64", "connections": "connections.i64"}.items()}
        self.hashers = {key: hashlib.sha256() for key in self.handles}
        self.buffers = {"books": np.empty((chunk_size, 2, keep_depth, 2), dtype="<f8"),
                        **{key: np.empty(chunk_size, dtype="<i8")
                           for key in ("times", "epochs", "connections")}}
        self.used, self.count = 0, 0

    def add(self, ts, epoch, connection, bids, asks, bid_prices, ask_prices):
        count = self.buffers["books"].shape[2]
        out = self.buffers["books"][self.used]
        bp, ap = bid_prices[-count:][::-1], ask_prices[:count]
        out[0, :, 0] = bp
        out[1, :, 0] = ap
        out[0, :, 1] = [bids[p] for p in bp]
        out[1, :, 1] = [asks[p] for p in ap]
        self.buffers["times"][self.used] = ts
        self.buffers["epochs"][self.used] = epoch
        self.buffers["connections"][self.used] = connection
        self.used += 1
        self.count += 1
        if self.used == len(self.buffers["times"]):
            self.flush()

    def flush(self):
        if self.used:
            for key, handle in self.handles.items():
                block = self.buffers[key][:self.used].tobytes(order="C")
                handle.write(block)
                self.hashers[key].update(block)
            self.used = 0

    def close(self):
        self.flush()
        for handle in self.handles.values():
            handle.close()
        return {key: {"filename": Path(handle.name).name,
                      "bytes": Path(handle.name).stat().st_size,
                      "sha256": self.hashers[key].hexdigest()}
                for key, handle in self.handles.items()}


def _reconstruct_books(path, day, symbol, writer, keep_depth, max_gap_ms):
    counters = Counter()
    start, end = _day_bounds(day)
    connection = int(day.replace("-", ""))
    epoch = connection * 1_000_000
    bids, asks, bid_prices, ask_prices = {}, {}, [], []
    active, last_u, last_ts, last_emitted_ts = False, None, None, None
    with zipfile.ZipFile(path) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        if len(entries) != 1 or not entries[0].filename.endswith(".data"):
            raise ValueError("Expected one orderbook archive member")
        with archive.open(entries[0]) as stream:
            for line in stream:
                counters["book_messages"] += 1
                try:
                    msg = _loads(line)
                    item = msg["data"]
                    if msg["topic"] != f"orderbook.200.{symbol}" or item["s"] != symbol:
                        raise ValueError("Archive contains a different orderbook symbol")
                    kind = msg["type"]
                    if kind not in {"snapshot", "delta"}:
                        raise ValueError("Unknown orderbook message type")
                    if "cts" not in msg or msg["cts"] is None:
                        counters["missing_matching_timestamp"] += 1
                        active = False
                        epoch += 1
                        continue
                    ts, update = _integer(msg["cts"]), _integer(item["u"])
                    if update < 1:
                        raise ValueError("Invalid update sequence")
                    if not start <= ts < end:
                        counters["book_messages_outside_utc_day"] += 1
                        # A small trailing spill is present in official daily archives.
                        # The adjacent day owns it; never silently combine file boundaries.
                        active = False
                        epoch += 1
                        continue
                    changes = (_levels(item["b"]), _levels(item["a"]))
                except (KeyError, TypeError, ValueError, OverflowError):
                    counters["malformed_book_messages"] += 1
                    active = False
                    epoch += 1
                    continue
                if kind == "snapshot":
                    counters["snapshots"] += 1
                    if last_emitted_ts is not None and ts < last_emitted_ts:
                        counters["reordered_snapshots"] += 1
                        active = False
                        epoch += 1
                        continue
                    # Even a repeated snapshot deliberately separates event windows.
                    epoch += 1
                    bids, asks, bid_prices, ask_prices = {}, {}, [], []
                    active, last_u, last_ts = True, None, None
                elif not active:
                    counters["delta_without_valid_snapshot"] += 1
                    continue
                if last_ts is not None:
                    if ts < last_ts or update <= last_u:
                        counters["reordered_or_duplicate_books"] += 1
                        active = False
                        epoch += 1
                        continue
                    if update != last_u + 1:
                        counters["update_sequence_gaps"] += 1
                        active = False
                        epoch += 1
                        continue
                    if ts - last_ts > 1000:
                        counters["book_gaps_over_1000ms"] += 1
                        counters["longest_book_gap_ms"] = max(counters["longest_book_gap_ms"], ts-last_ts)
                    if max_gap_ms and ts - last_ts > max_gap_ms:
                        counters["book_gaps_over_max_age"] += 1
                        epoch += 1
                    if ts == last_ts:
                        counters["same_millisecond_book_updates"] += 1
                for state, prices, update_rows in ((bids, bid_prices, changes[0]),
                                                    (asks, ask_prices, changes[1])):
                    for price, quantity in update_rows:
                        if quantity == 0:
                            if price in state:
                                del state[price]
                                prices.pop(bisect.bisect_left(prices, price))
                        else:
                            if price not in state:
                                bisect.insort(prices, price)
                            state[price] = quantity
                last_u, last_ts = update, ts
                if len(bids) < keep_depth or len(asks) < keep_depth:
                    counters["shallow_books"] += 1
                    active = False
                    epoch += 1
                    continue
                if bid_prices[-1] >= ask_prices[0]:
                    counters["crossed_or_locked_books"] += 1
                    active = False
                    epoch += 1
                    continue
                counters["max_internal_bid_levels"] = max(counters["max_internal_bid_levels"], len(bids))
                counters["max_internal_ask_levels"] = max(counters["max_internal_ask_levels"], len(asks))
                writer.add(ts, epoch, connection, bids, asks, bid_prices, ask_prices)
                last_emitted_ts = ts
    counters["valid_books"] = writer.count
    if not writer.count:
        raise ValueError(f"No valid orderbook observations for {day}: {dict(counters)}")
    return counters


def _read_trades(path, day):
    counters = Counter()
    start, end = _day_bounds(day)
    connection = int(day.replace("-", ""))
    seen, rows = {}, []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "timestamp", "price", "volume", "side", "rpi"]:
            raise ValueError("Unexpected public spot trade CSV header")
        for row in reader:
            counters["trade_rows"] += 1
            try:
                trade_id = row["id"]
                if not trade_id:
                    raise ValueError("Missing trade id")
                ts = _integer(row["timestamp"])
                price, quantity = float(row["price"]), float(row["volume"])
                side, rpi = row["side"].lower(), _integer(row["rpi"])
                if side not in {"buy", "sell"} or rpi not in {0, 1}:
                    raise ValueError("Invalid side or RPI flag")
                if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity <= 0:
                    raise ValueError("Invalid trade price or quantity")
            except (KeyError, TypeError, ValueError, OverflowError):
                counters["invalid_trades"] += 1
                continue
            identity = (ts, price, quantity, side, rpi)
            if trade_id in seen:
                if identity != seen[trade_id]:
                    raise ValueError(f"Conflicting rows use the same public trade id: {trade_id}")
                counters["duplicate_trade_ids"] += 1
                continue
            seen[trade_id] = identity
            if not start <= ts < end:
                counters["trades_outside_utc_day"] += 1
                continue
            if rpi:
                counters["rpi_trades_excluded"] += 1
                continue
            # Identical fills with DIFFERENT exchange IDs remain separate trades.
            rows.append((ts, connection, side.capitalize(), quantity, price))
    if not rows:
        raise ValueError("No valid non-RPI public trades")
    frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    counters["reordered_trade_timestamps"] = int((frame.ts.diff().dropna() < 0).sum())
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    counters["valid_trades"] = len(frame)
    return frame, counters


def load_day(cache_dir, mmap=True, verify_hashes=False):
    """Open a completed per-day cache.  No pickle or executable payload is read."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["cache_version"] != CACHE_VERSION:
        raise ValueError("Incompatible cache version")
    count, depth = manifest["book_count"], manifest["keep_depth"]
    data = {}
    for key, spec in manifest["arrays"].items():
        path = cache_dir / spec["filename"]
        dtype = "<f8" if key == "books" else "<i8"
        shape = (count, 2, depth, 2) if key == "books" else (count,)
        if path.stat().st_size != spec["bytes"] or path.stat().st_size != math.prod(shape)*8:
            raise ValueError(f"Truncated cache array: {key}")
        if verify_hashes and sha256(path) != spec["sha256"]:
            raise ValueError(f"Cache array checksum mismatch: {key}")
        data[key] = np.memmap(path, dtype=dtype, mode="r", shape=shape) if mmap else np.fromfile(path, dtype=dtype).reshape(shape)
    trade_path = cache_dir / "trades.csv.gz"
    if sha256(trade_path) != manifest["trades_sha256"]:
        raise ValueError("Cache trade checksum mismatch")
    data["trades"] = pd.read_csv(trade_path, dtype={"ts": "int64", "connection": "int64",
                                                  "quantity": "float64", "price": "float64"})
    if list(data["trades"].columns) != TRADE_COLUMNS or len(data["trades"]) != manifest["trade_count"]:
        raise ValueError("Cache trades have unexpected columns or row count")
    data.update(quality=manifest["quality"], manifest=manifest, cache_dir=str(cache_dir))
    return data


def prepare_day(root, day, symbol="ETHUSDT", cache_root=None, keep_depth=50,
                max_gap_ms=1000, mmap=True):
    """Verify sources, reconstruct one UTC day, and return analysis-compatible data.

    The disk cache is immutable once complete.  Partial caches use a temporary
    sibling directory and are removed after failures.  Per-process book buffers
    take a few MB; no list of all daily book tensors is retained in memory.
    """
    day, symbol = _identifiers(day, symbol)
    if isinstance(keep_depth, bool) or not 5 <= keep_depth <= 200:
        raise ValueError("keep_depth must be between 5 and 200")
    if not isinstance(keep_depth, int) or not isinstance(max_gap_ms, int) or max_gap_ms < 0:
        raise ValueError("Depth must be an integer; maximum gap must be a nonnegative integer")
    root = Path(root)
    source_manifest = download_day(root, day, symbol)
    cache_root = Path(cache_root) if cache_root is not None else root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / f"{day}-{symbol}-depth{keep_depth}-gap{max_gap_ms}-v{CACHE_VERSION}"
    source_hashes = {key: entry["sha256"] for key, entry in source_manifest["files"].items()}
    if cache_dir.exists():
        loaded = load_day(cache_dir, mmap=mmap)
        if loaded["manifest"]["source_sha256"] != source_hashes:
            raise ValueError("Source hashes differ from existing cache; refusing replacement")
        return loaded
    temporary = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".partial-", dir=cache_root))
    writer = None
    try:
        writer = _BookWriter(temporary, keep_depth)
        quality = _reconstruct_books(root / source_manifest["files"]["orderbook"]["filename"],
                                     day, symbol, writer, keep_depth, max_gap_ms)
        arrays = writer.close()
        writer = None
        trades, trade_quality = _read_trades(root / source_manifest["files"]["trades"]["filename"], day)
        quality.update(trade_quality)
        trade_path = temporary / "trades.csv.gz"
        trades.to_csv(trade_path, index=False, compression={"method": "gzip", "mtime": 0})
        manifest = {"cache_version": CACHE_VERSION, "venue": "Bybit", "market": "spot",
                    "symbol": symbol, "day_utc": day, "keep_depth": keep_depth,
                    "source_depth": 200, "max_gap_ms": max_gap_ms,
                    "clock": "matching_engine_cts", "source_sha256": source_hashes,
                    "sources": source_manifest["files"], "quality": dict(quality),
                    "book_count": quality["valid_books"], "trade_count": len(trades),
                    "arrays": arrays, "trades_sha256": sha256(trade_path),
                    "boundary_policy": ("Independent UTC days; new epoch on snapshot; sequence loss requires snapshot; "
                                        + (f"new epoch on gaps >{max_gap_ms} ms" if max_gap_ms else "continuous update sequence can span unchanged-book intervals")),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_json_exclusive(temporary / "manifest.json", manifest)
        # rename refuses an existing nonempty completed cache.
        temporary.rename(cache_dir)
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_day(cache_dir, mmap=mmap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--day", required=True, help="UTC date YYYY-MM-DD")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--prepare", action="store_true", help="Reconstruct mmap cache after download")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--keep-depth", type=int, default=50)
    parser.add_argument("--max-gap-ms", type=int, default=1000,
                        help="Start new epoch after this gap; 0 disables gap-based epoch cuts")
    args = parser.parse_args()
    if args.prepare:
        data = prepare_day(args.root, args.day, args.symbol, args.cache_root,
                           args.keep_depth, args.max_gap_ms)
        print(json.dumps({"cache_dir": data["cache_dir"], "quality": data["quality"]}, indent=2))
    else:
        print(json.dumps(download_day(args.root, args.day, args.symbol), indent=2))


if __name__ == "__main__":
    main()
