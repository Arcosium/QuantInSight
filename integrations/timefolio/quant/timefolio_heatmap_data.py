"""Read-only KRX research extraction. Never imports a broker or writes source DBs."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "vault/ArcTrade/timefolio_heatmap/20260928"
KRX = Path.home() / "vault/CryptoBars/data/KRX"
START, END = "20250908", "20260923"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def read_calendar(db=KRX / "bars_ohlc.db"):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        return [x[0] for x in con.execute(
            "SELECT DISTINCT day FROM done WHERE day>=? AND day<=? ORDER BY day", (START, END))]


def aggregate_minutes(raw):
    """15-min regular-session bins; closing auction joins the final bin.

    A separate 09:05--09:34 execution window is strictly after yesterday's signal.
    OHLC typical-price VWAP is explicitly a proxy, not a tick VWAP or order-book fill.
    Missing regular minutes are not forward/backward filled.
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = raw.copy()
    d["date"] = d.ts.str[:8]
    d["minute"] = d.ts.str[8:10].astype(int) * 60 + d.ts.str[10:12].astype(int)
    valid = (d[["o", "h", "l", "c"]] > 0).all(axis=1) & (d.v >= 0)
    valid &= (d.h >= d[["o", "l", "c"]].max(axis=1)) & (d.l <= d[["o", "h", "c"]].min(axis=1))
    d = d.loc[valid & d.minute.between(540, 990)].copy()
    d["amount_proxy"] = (d.h + d.l + d.c) / 3 * d.v
    daily = d.groupby("date", sort=True).agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
        v=("v", "sum"), amount_proxy=("amount_proxy", "sum"),
        count=("ts", "size"), first_minute=("minute", "min"), last_minute=("minute", "max"))
    ex = d.loc[d.minute.between(545, 574)].groupby("date").agg(
        exec_amount=("amount_proxy", "sum"), exec_volume=("v", "sum"),
        exec_high=("h", "max"), exec_low=("l", "min"), exec_count=("ts", "size"))
    daily = daily.join(ex)
    daily["exec_price"] = daily.exec_amount / daily.exec_volume.replace(0, np.nan)
    daily["regular"] = (daily.first_minute == 540) & (daily.last_minute == 930) & (daily["count"] >= 300)
    # Extended/delayed-open sessions remain in daily marks, but cannot generate fills/signals.
    daily.loc[~daily.regular, ["exec_price", "exec_volume"]] = np.nan
    d = d.loc[(d.minute < 920) | (d.minute == 930)].copy()
    d["slot"] = np.minimum((d.minute - 540) // 15, 25)
    bars = d.groupby(["date", "slot"], sort=True).agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
        v=("v", "sum"), count=("ts", "size"))
    return bars.reset_index(), daily.reset_index()


def build_bars(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    shards = root / "bar_shards"
    shards.mkdir(exist_ok=True)
    with (KRX / "universe_all.csv").open() as f:
        universe = list(csv.DictReader(f))
    # Include archived tickers absent from today's list; their metadata can remain unknown.
    with sqlite3.connect(f"file:{KRX / 'bars_ohlc.db'}?mode=ro", uri=True) as con:
        codes = sorted(set(x[0] for x in con.execute("SELECT DISTINCT code FROM done")))
    atomic_json(root / "universe_snapshot.json", universe)
    atomic_json(root / "calendar.json", read_calendar())
    began = time.monotonic()
    for n, code in enumerate(codes):
        bp, dp = shards / f"{code}.parquet", shards / f"{code}.daily.parquet"
        if bp.exists() and dp.exists():
            continue
        # Short snapshots avoid holding back a live collector's WAL checkpoint.
        with sqlite3.connect(f"file:{KRX / 'bars_ohlc.db'}?mode=ro", uri=True, timeout=30) as con:
            con.execute("PRAGMA query_only=ON")
            raw = pd.read_sql_query(
                "SELECT ts,o,h,l,c,v FROM bars WHERE code=? AND ts>=? AND ts<? ORDER BY ts",
                con, params=(code, START + "000000", END + "235959"))
        bars, daily = aggregate_minutes(raw)
        bars.to_parquet(bp, index=False)
        daily.to_parquet(dp, index=False)
        if n % 100 == 0 or n == len(codes) - 1:
            print(json.dumps({"stage": "minute_aggregation", "completed": n + 1,
                              "codes": len(codes), "seconds": round(time.monotonic() - began)}), flush=True)
    parts = []
    for f in sorted(shards.glob("*.daily.parquet")):
        d = pd.read_parquet(f)
        if not d.empty:
            d["code"] = f.name.split(".")[0]
            parts.append(d)
    daily = pd.concat(parts, ignore_index=True)
    daily.to_parquet(root / "daily_minutes.parquet", index=False)
    print(json.dumps({"stage": "minute_aggregation_done", "rows": len(daily),
                      "codes": int(daily.code.nunique())}), flush=True)


def build_krx_metadata(root=ROOT):
    """Point-in-time daily market cap, listed shares and actual exchange turnover."""
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "krx_daily"
    dest.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from pykrx import stock
    rename = {"종가": "krx_close", "시가총액": "market_cap", "거래량": "krx_volume",
              "거래대금": "traded_value", "상장주식수": "listed_shares"}
    dates = read_calendar()
    for n, day in enumerate(dates):
        p = dest / f"{day}.parquet"
        if p.exists():
            continue
        frame = None
        for attempt in range(3):
            try:
                # Third-party authentication chatter must never reach research logs.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    frame = stock.get_market_cap_by_ticker(day, market="ALL")
                if not frame.empty and set(rename).issubset(frame.columns):
                    break
            except Exception:
                frame = None
            time.sleep(1 + attempt)
        if frame is None or frame.empty:
            raise RuntimeError(f"Historical KRX metadata unavailable for {day}; do not backfill from future.")
        frame = frame.rename(columns=rename).rename_axis("code").reset_index()
        frame["date"] = day
        frame.to_parquet(p, index=False)
        if n % 10 == 0 or n == len(dates) - 1:
            print(json.dumps({"stage": "historical_krx", "completed": n + 1, "dates": len(dates)}), flush=True)
        time.sleep(.15)
    full = pd.concat([pd.read_parquet(dest / f"{day}.parquet") for day in dates], ignore_index=True)
    full.to_parquet(root / "historical_krx.parquet", index=False)
    print(json.dumps({"stage": "historical_krx_done", "rows": len(full)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bars", "metadata", "all"])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.command in ("bars", "all"):
        build_bars(args.root)
    if args.command in ("metadata", "all"):
        build_krx_metadata(args.root)
