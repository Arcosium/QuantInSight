from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .sessions import KST


def _now_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _path(data_dir: Path, uid: int = 1) -> Path:
    p = data_dir / str(uid)
    p.mkdir(parents=True, exist_ok=True)
    return p / "ledger.json"


def load(data_dir: Path, uid: int = 1) -> dict[str, Any] | None:
    p = _path(data_dir, uid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(data_dir: Path, ledger: dict[str, Any], uid: int = 1) -> None:
    _path(data_dir, uid).write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")


def seed(data_dir: Path, *, cash_krw: float, positions: dict[str, dict[str, Any]] | None = None, uid: int = 1) -> dict[str, Any]:
    ledger = {
        "version": 1,
        "seeded_at": _now_str(),
        "seed_source": "timefolio_mock",
        "cash_krw": float(cash_krw or 0.0),
        "cash_usd": 0.0,
        "positions": positions or {},
        "fills": [],
        "degraded_fills": 0,
    }
    save(data_dir, ledger, uid)
    return ledger


def apply_fill(
    data_dir: Path,
    *,
    ticker: str,
    side: str,
    qty: int,
    price: float,
    uid: int = 1,
    note: str = "",
) -> bool:
    ledger = load(data_dir, uid)
    if ledger is None:
        return False
    ticker = str(ticker or "").strip()
    side = str(side or "").lower()
    qty = int(qty or 0)
    price = float(price or 0.0)
    if not ticker or side not in {"buy", "sell"} or qty <= 0 or price <= 0:
        ledger["degraded_fills"] = int(ledger.get("degraded_fills") or 0) + 1
        save(data_dir, ledger, uid)
        return False

    positions = ledger.setdefault("positions", {})
    pos = positions.get(ticker)
    realized = None
    if side == "buy":
        ledger["cash_krw"] = float(ledger.get("cash_krw") or 0.0) - price * qty
        if pos:
            old_qty = int(pos.get("qty") or 0)
            old_avg = float(pos.get("avg_cost") or 0.0)
            new_qty = old_qty + qty
            pos["avg_cost"] = ((old_avg * old_qty + price * qty) / new_qty) if new_qty > 0 else price
            pos["qty"] = new_qty
        else:
            positions[ticker] = {"qty": qty, "avg_cost": price, "ccy": "KRW", "last_price": price}
    else:
        if not pos:
            ledger["degraded_fills"] = int(ledger.get("degraded_fills") or 0) + 1
            save(data_dir, ledger, uid)
            return False
        avg = float(pos.get("avg_cost") or 0.0)
        sell_qty = min(qty, int(pos.get("qty") or 0))
        ledger["cash_krw"] = float(ledger.get("cash_krw") or 0.0) + price * sell_qty
        pos["qty"] = int(pos.get("qty") or 0) - sell_qty
        pos["last_price"] = price
        if avg > 0:
            realized = (price - avg) * sell_qty
        if pos["qty"] <= 0:
            positions.pop(ticker, None)
        qty = sell_qty

    fill = {
        "ts": _now_str(),
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "price": price,
        "ccy": "KRW",
        "fee": 0.0,
        "approx_price": False,
        "note": note[:120],
    }
    if realized is not None:
        fill["realized"] = round(realized, 4)
    ledger.setdefault("fills", []).append(fill)
    ledger["fills"] = ledger["fills"][-2000:]
    save(data_dir, ledger, uid)
    return True
