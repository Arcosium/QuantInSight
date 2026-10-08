from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta
from typing import Any

import requests

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_daily_ohlcv(code: str, *, pages: int = 3, timeout: float = 8.0) -> list[dict[str, Any]]:
    # sise_day.nhn 은 2026-09 부터 410 Gone → siseJson 일봉. pages 는 옛 페이지(10거래일) 단위로 유지한다.
    code = str(code).strip().zfill(6)
    end = datetime.now().date()
    start = end - timedelta(days=max(1, int(pages)) * 16)
    res = requests.get(
        "https://api.finance.naver.com/siseJson.naver",
        params={"symbol": code, "requestType": 1, "timeframe": "day",
                "startTime": start.strftime("%Y%m%d"), "endTime": end.strftime("%Y%m%d")},
        headers=_HEADERS, timeout=timeout,
    )
    res.raise_for_status()
    rows: list[dict[str, Any]] = []
    for r in ast.literal_eval(res.text.strip())[1:]:  # 첫 줄은 머리글
        if len(r) < 6:
            continue
        rows.append({
            "date": datetime.strptime(str(r[0]), "%Y%m%d").date().isoformat(),
            "open": int(r[1]), "high": int(r[2]), "low": int(r[3]), "close": int(r[4]), "volume": int(r[5]),
        })
    return rows


def fetch_security_meta(code: str, *, stored: dict[str, Any] | None = None, timeout: float = 8.0) -> dict[str, Any]:
    code = str(code).strip().zfill(6)
    meta: dict[str, Any] = dict(stored or {})
    meta["ticker"] = code
    daily = fetch_daily_ohlcv(code, pages=2, timeout=timeout)
    if daily:
        recent = daily[-5:]
        meta["last_price"] = float(daily[-1]["close"])
        meta["listed_business_days"] = max(int(meta.get("listed_business_days") or 0), len(daily))
        if recent:
            meta["avg_5d_trading_value_krw"] = sum(float(r["close"]) * float(r["volume"]) for r in recent) / len(recent)

    # main.naver 는 2026-09 부터 stock.naver.com SPA 로 302 → 모바일 JSON API 사용.
    try:
        base = f"https://m.stock.naver.com/api/stock/{code}"
        basic = requests.get(f"{base}/basic", headers=_HEADERS, timeout=timeout).json()
        if basic.get("stockName") and not meta.get("name"):
            meta["name"] = basic["stockName"]
        if not meta.get("market"):
            ex = str(basic.get("stockExchangeName") or "").upper()
            if ex in ("KOSPI", "KOSDAQ"):
                meta["market"] = ex
        flags = list(meta.get("flags") or []) if not isinstance(meta.get("flags"), str) else [x.strip() for x in meta.get("flags", "").split(",") if x.strip()]
        # ponytail: 투자주의·경고·위험·관리종목은 이 API 에 없어 거래정지만 잡는다 — 필요하면 KIS 종목상태코드로 보강
        if (basic.get("tradeStopType") or {}).get("name") not in (None, "TRADING") and "거래정지" not in flags:
            flags.append("거래정지")
        meta["flags"] = flags
        if not meta.get("market_cap_krw"):
            info = requests.get(f"{base}/integration", headers=_HEADERS, timeout=timeout).json()
            for item in info.get("totalInfos") or []:
                if item.get("code") == "marketValue":
                    cap = _parse_eok(str(item.get("value") or ""))
                    if cap:
                        meta["market_cap_krw"] = cap
        if meta.get("is_common_stock") is None:
            nm = str(meta.get("name") or "")
            meta["is_common_stock"] = not any(x in nm.upper() for x in ("ETF", "ETN", "스팩", "우B")) and not nm.endswith("우")
    except Exception:
        pass
    return meta


def _parse_eok(text: str) -> float | None:
    """'14조 6,925억' → 원."""
    jo = re.search(r"([\d,]+)\s*조", text)
    eok = re.search(r"([\d,]+)\s*억", text)
    if not jo and not eok:
        return None
    total = (float(jo.group(1).replace(",", "")) * 10_000 if jo else 0.0) + (float(eok.group(1).replace(",", "")) if eok else 0.0)
    return total * 100_000_000
