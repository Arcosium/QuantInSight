"""Daily OHLCV alongside the shared KRX/NXT/crypto archive; no account data."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from arcmarket.systematic import clean_bars

ROOT = Path(os.getenv("ARCTRADE_MINUTE_DATA_DIR", Path(__file__).resolve().parents[2]/"CryptoBars"/"data"))
DAILY = ROOT/"KRX"/"daily_cnn"
US_DAILY = ROOT/"USA"/"daily_policy"
US_POLICY_UNIVERSE = tuple("AAPL MSFT AMZN GOOGL META NVDA AVGO AMD ORCL CRM ADBE JPM BAC GS V MA XOM CVX COP UNH JNJ LLY ABBV MRK TMO PG KO PEP COST WMT HD CAT GE HON RTX UPS NEE DUK AMT PLD".split())


def universe():
    with (ROOT/"KRX"/"universe.csv").open() as f:
        return {r["code"]: r.get("name", r["code"]) for r in csv.DictReader(f)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        tmp = f.name
    os.replace(tmp, path)


def fetch(code, *, destination=None):
    r = requests.get("https://fchart.stock.naver.com/sise.nhn", params={"symbol": code,
        "timeframe": "day", "count": "1000", "requestType": "0"}, timeout=20)
    r.raise_for_status()
    rows = [x.get("data", "").split("|") for x in ET.fromstring(r.text).findall(".//item")]
    d = pd.DataFrame([x for x in rows if len(x)==6], columns=["date", "open", "high", "low", "close", "volume"])
    d["date"] = pd.to_datetime(d.date, format="%Y%m%d")
    d = clean_bars(d)
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    # Today's provisional candle never enters training or decisions.
    d = d[d.index < today]
    if len(d) < 61 or (today-d.index[-1]).days > 7:
        raise ValueError("insufficient_or_stale_bars")
    target = Path(destination) if destination is not None else DAILY/f"{code}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".parquet", delete=False) as f:
        tmp = f.name
    try:
        d.to_parquet(tmp)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return str(d.index[-1].date())


def refresh():
    codes = universe()
    def one(c):
        try:
            last = fetch(c)
            return c, last, None
        except Exception as e:
            return c, None, type(e).__name__
        finally:
            time.sleep(.12)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, codes))
    status = {"updated_at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "NAVER daily chart",
              "adjustment": "provider chart basis; dividends not credited; corporate actions not independently audited",
              "requested": len(codes), "ok": sum(x[1] is not None for x in rows),
              "errors": {c: e for c, _, e in rows if e}, "as_of": max((x[1] for x in rows if x[1]), default=None)}
    atomic_json(DAILY/"status.json", status)
    return status


def load():
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    codes = universe()
    return {p.stem: d[d.index < today] for p in sorted(DAILY.glob("*.parquet"))
            if p.stem in codes for d in [pd.read_parquet(p)]}


def refresh_us_policy():
    """Fixed liquid-stock research universe. US data never trains the KRX CNN."""
    import arcmarket
    US_DAILY.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="America/New_York")
    cutoff = now.tz_localize(None).normalize() + (pd.Timedelta(days=1) if now.hour >= 16 else pd.Timedelta(0))
    errors, ends = {}, []
    for code in US_POLICY_UNIVERSE:
        try:
            d = arcmarket.us_daily(code, days=1100, adjusted=True)
            if d is None:
                raise ValueError("no_data")
            d = clean_bars(d); d = d[d.index < cutoff]
            if len(d) < 61:
                raise ValueError("short_history")
            with tempfile.NamedTemporaryFile(dir=US_DAILY, suffix=".parquet", delete=False) as f:
                tmp = f.name
            try:
                d.to_parquet(tmp); os.replace(tmp, US_DAILY/f"{code}.parquet")
            finally:
                Path(tmp).unlink(missing_ok=True)
            ends.append(str(d.index[-1].date()))
        except Exception as e:
            errors[code] = type(e).__name__
    status = {"updated_at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "arcmarket/yfinance adjusted daily",
              "codes": list(US_POLICY_UNIVERSE), "requested": len(US_POLICY_UNIVERSE), "ok": len(ends),
              "as_of": max(ends, default=None), "errors": errors}
    atomic_json(US_DAILY/"status.json", status)
    return status


if __name__ == "__main__":
    print(json.dumps(refresh(), ensure_ascii=False))
