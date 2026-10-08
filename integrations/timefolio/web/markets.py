"""Shared market catalogue, venue-aware bars and cost-aware paper research."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

ROOT = Path(os.getenv("ARCTRADE_MINUTE_DATA_DIR", Path(__file__).resolve().parents[2] / "CryptoBars" / "data"))
KST = ZoneInfo("Asia/Seoul")
MARKETS = {
    "KRX": {"name": "한국거래소", "currency": "KRW", "interval": "1분", "cost_pct": .35,
            "source": "네이버 금융 · 기존 KRX 종가 아카이브"},
    "NXT": {"name": "넥스트레이드", "currency": "KRW", "interval": "1분", "cost_pct": .35,
            "source": "한국투자증권 · NXT 전용 시세"},
    "USA": {"name": "미국 주식", "currency": "USD", "interval": "일봉", "cost_pct": .70,
            "source": "공용 미국 주식 일봉 · 요청 시 갱신"},
    "CRYPTO": {"name": "크립토", "currency": "USD / USDT / USDC", "interval": "1분", "cost_pct": .20,
               "source": "CryptoBars · 거래대금 기준 대표 거래소"},
}
US_SYMBOLS = ("SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA")
router = APIRouter(prefix="/api/markets")


def validate(market, symbol=None):
    if market not in MARKETS:
        raise ValueError("지원하지 않는 마켓입니다")
    if symbol is not None:
        pattern = r"[0-9]{6}" if market in ("KRX", "NXT") else r"[\w.-]{1,40}"
        if not re.fullmatch(pattern, symbol) or ".." in symbol:
            raise ValueError("종목 코드를 확인하세요")


def _csv(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def symbols(market):
    validate(market)
    if market in ("KRX", "NXT"):
        rows = [{"symbol": r["code"], "name": r["name"]} for r in _csv(ROOT / "KRX/universe.csv")]
        if market == "NXT":
            path = ROOT / "NXT/bars.db"
            if not path.exists():
                return []
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
                available = {r[0] for r in c.execute("SELECT DISTINCT code FROM bars")}
            rows = [r for r in rows if r["symbol"] in available]
        return rows
    if market == "USA":
        return [{"symbol": s, "name": s} for s in US_SYMBOLS]
    return [{"symbol": r["base"], "name": r["base"], "venue": r["venue"]}
            for r in _csv(ROOT / "universe.csv")]


def _local_bars(market, symbol, limit):
    path = ROOT / market / "bars.db"
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10) as c:
        if market == "KRX":
            rows = c.execute("SELECT ts,close,vol_cum FROM bars WHERE code=? ORDER BY ts DESC LIMIT ?",
                             (symbol, limit)).fetchall()
            # The legacy source supplies only close and cumulative volume.
            return [{"time": int(datetime.strptime(ts, "%Y%m%d%H%M").replace(tzinfo=KST).timestamp()),
                     "close": close, "volume_cumulative": vol, "source": "Naver:KRX"}
                    for ts,close,vol in reversed(rows)]
        rows = c.execute("SELECT ts,open,high,low,close,volume,source FROM bars WHERE code=? ORDER BY ts DESC LIMIT ?",
                         (symbol, limit)).fetchall()
    return [{"time": int(datetime.strptime(ts,"%Y%m%d%H%M").replace(tzinfo=KST).timestamp()),
             "open": o, "high": h, "low": l, "close": close, "volume": v, "source": src}
            for ts,o,h,l,close,v,src in reversed(rows)]


def _us_bars(symbol, limit):
    path = ROOT / "USA" / f"{symbol}.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or datetime.now().timestamp() - path.stat().st_mtime > 3600:
        try:
            import arcmarket
            df = arcmarket.us_daily(symbol, days=1100, adjusted=True)
            if df is not None and not df.empty:
                df = df.reset_index() if "date" not in df.columns else df
                df.columns = [str(x).lower() for x in df.columns]
                if "date" not in df and "datetime" in df:
                    df = df.rename(columns={"datetime": "date"})
                if "date" in df and "close" in df:
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
                        tmp = Path(f.name)
                    try:
                        df.to_csv(tmp, index=False); tmp.replace(path)
                    finally:
                        tmp.unlink(missing_ok=True)
        except Exception:
            if not path.exists():
                raise ValueError("미국 일봉을 가져오지 못했습니다. 잠시 후 다시 조회하세요")
    if not path.exists():
        return []
    df = pd.read_csv(path).tail(limit)
    out = []
    today = datetime.now(ZoneInfo("America/New_York")).date()
    for r in df.to_dict("records"):
        date = pd.Timestamp(r["date"]).date()
        if date >= today:
            continue
        out.append({"time": int(datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc).timestamp()),
                    **{k: float(r[k]) for k in ("open","high","low","close","volume") if k in r and pd.notna(r[k])},
                    "source": "arcmarket:USA:adjusted"})
    return out


def _crypto_bars(symbol, limit):
    import pyarrow.parquet as pq
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=14)
    cutoff = int(now.timestamp() // 60 * 60000)
    files = list((ROOT / "history" / f"base={symbol}").glob("*.parquet"))
    # Monthly history partitions: scan only months intersecting the window.
    files = [p for p in files if p.stem[-7:] >= since.strftime("%Y-%m") or since.strftime("%Y%m") in p.name]
    # Use folder dates for live partitions; never scan the entire 37 GB archive.
    for folder in (ROOT / "bars").glob("date=*"):
        if folder.name[5:] >= since.strftime("%Y-%m-%d"):
            files.extend(sorted(folder.glob("*.parquet")))
    records = {}
    for p in files:
        t = pq.read_table(p, filters=[("base","=",symbol), ("ts",">=",int(since.timestamp()*1000)), ("ts","<",cutoff)],
                          columns=["ts","base","open","high","low","close","volume","venue"])
        for r in t.to_pylist():
            records[r["ts"]] = {"time":r["ts"]//1000, **{k:r[k] for k in ("open","high","low","close","volume")},
                                 "source":r["venue"]}
    return [records[t] for t in sorted(records)[-limit:]]


def bars(market, symbol, limit=1000):
    validate(market, symbol)
    if market in ("KRX", "NXT"):
        rows = _local_bars(market, symbol, limit)
    elif market == "USA":
        rows = _us_bars(symbol, limit)
    else:
        rows = _crypto_bars(symbol, limit)
    now = int(datetime.now(timezone.utc).timestamp())
    return [r for r in rows if r["time"] < now - (now % 60) and math.isfinite(r["close"]) and r["close"] > 0]


def trend_backtest(rows, *, fast=20, slow=60, cost_pct=.35):
    """Signals use previous close; executions use the next bar open/close.

    Closing-only archives execute at the following close. Both execution
    paths lag by one observation and include entry/exit costs. No optimization.
    """
    if not 2 <= fast < slow <= 500 or not math.isfinite(cost_pct) or not 0 <= cost_pct <= 10:
        raise ValueError("기간은 2 ≤ 단기 < 장기 ≤ 500, 왕복비용은 0~10%로 입력하세요")
    if len(rows) < slow + 30:
        raise ValueError(f"완성 봉이 {slow+30}개 이상 필요합니다 (현재 {len(rows)}개)")
    close = pd.Series([r["close"] for r in rows])
    signals = (close.rolling(fast).mean() > close.rolling(slow).mean()).tolist()
    fee = cost_pct / 200
    cash, qty, trades = 1.0, 0.0, 0
    curve = []
    split = max(slow+1, int(len(rows)*.7))
    for i in range(slow, len(rows)):
        price = float(rows[i].get("open") or rows[i]["close"])
        want = signals[i-1]
        if want and not qty:
            qty = cash / (price*(1+fee)); cash = 0; trades += 1
        elif not want and qty:
            cash = qty*price*(1-fee); qty = 0; trades += 1
        nav = cash + qty*rows[i]["close"]*(1-fee)
        curve.append({"time":rows[i]["time"],"equity":nav,"holdout":i >= split})
    values = pd.Series([1.0] + [r["equity"] for r in curve])
    test = [r["equity"] for r in curve if r["holdout"]]
    base = curve[max(0,split-slow-1)]["equity"]
    buy_price = float(rows[slow].get("open") or rows[slow]["close"])
    return {"return_pct":(values.iloc[-1]-1)*100,
            "max_drawdown_pct":float((values/values.cummax()-1).min()*100),
            "holdout_return_pct":(test[-1]/base-1)*100,
            "benchmark_return_pct":(rows[-1]["close"]*(1-fee)/(buy_price*(1+fee))-1)*100,
            "trades":trades,"bars":len(rows),"cost_pct":cost_pct,"curve":curve,
            "execution":"다음 봉 시가" if all("open" in r for r in rows) else "다음 봉 종가",
            "note":"비용은 가정값입니다. 후반 30%를 따로 표시하며 매개변수를 반복 선택하면 독립 검증이 아닙니다."}


@router.get("")
def catalogue():
    try:
        status = json.loads((ROOT / "equities_status.json").read_text())
    except (OSError, ValueError):
        status = {}
    return {"markets":[{"id":k,**v,"collection":status.get(k,{})} for k,v in MARKETS.items()]}


@router.get("/{market}/symbols")
def get_symbols(market: str, q: str = ""):
    try:
        result = symbols(market)
        return {"symbols":[r for r in result if q.lower() in (r["symbol"]+r["name"]).lower()][:250], "total":len(result)}
    except ValueError as e:
        raise HTTPException(400,str(e))


@router.get("/{market}/bars")
def get_bars(market: str, symbol: str, limit: int = Query(1000,ge=50,le=20000)):
    try:
        return {"market":market,"symbol":symbol,"bars":bars(market,symbol,limit),"meta":MARKETS[market]}
    except (ValueError, KeyError) as e:
        raise HTTPException(400,str(e))


@router.post("/{market}/backtest")
def backtest(market: str, body: dict):
    try:
        validate(market)
        symbol = str(body.get("symbol", ""))
        rows = bars(market,symbol,20000 if market != "USA" else 1000)
        return {"market":market,"symbol":symbol,**trend_backtest(rows,
            fast=int(body.get("fast",20)),slow=int(body.get("slow",60)),
            cost_pct=float(body.get("cost_pct",MARKETS[market]["cost_pct"])))}
    except (ValueError,TypeError,OverflowError) as e:
        raise HTTPException(400,str(e))
