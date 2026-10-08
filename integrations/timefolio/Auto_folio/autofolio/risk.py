from __future__ import annotations

import json
import re
from typing import Any

from . import config
from .sessions import is_kr_code


def _orders_from_any(order_json: Any) -> list[dict[str, Any]]:
    data = order_json
    if isinstance(order_json, str):
        txt = order_json.strip()
        match = re.search(r"\{.*\}", txt, re.S)
        data = json.loads(match.group(0) if match else txt)
    if isinstance(data, dict) and "orders" in data:
        return [x for x in data.get("orders") or [] if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def validate_order_draft(
    order_json: Any,
    *,
    buying_power: dict[str, Any],
    price_map: dict[str, float],
) -> dict[str, Any]:
    """QuantInSight-style deterministic KR risk validation.

    This mirrors the KR-only branch of `agents.guardrails.validate_order_draft`:
    buys require price, healthy balance, cash buffer, MDD, single-name cap and
    cycle budget; sells are risk-reducing and pass structural validation.
    """
    try:
        orders = _orders_from_any(order_json)
    except Exception:
        return {"approved": False, "results": [], "report": "주문 JSON 파싱 실패 — 실행 보류."}
    if not orders:
        return {"approved": False, "results": [], "report": "검증할 주문 없음 — 실행 없음."}

    cash = float(buying_power.get("cash") or 0.0)
    total = float(buying_power.get("total_eval") or 0.0) or cash
    pnl_ratio = float(buying_power.get("pnl_ratio") or 0.0)
    bp_ok = bool(buying_power.get("ok"))
    spent = 0.0
    results: list[dict[str, Any]] = []

    for raw in orders:
        ticker = str(raw.get("ticker") or "").strip()
        side = str(raw.get("side") or "buy").lower()
        reason = str(raw.get("reason") or "")
        issues: list[str] = []
        try:
            qty = int(float(raw.get("qty") or 0))
        except (TypeError, ValueError):
            qty = 0

        if not ticker:
            issues.append("종목코드(ticker) 미지정")
        elif not is_kr_code(ticker):
            issues.append("국장 모의 계정은 6자리 국내 종목코드만 허용")
        if side not in {"buy", "sell"}:
            issues.append(f"지원하지 않는 주문 방향({side})")
        if qty <= 0:
            issues.append(f"주문 수량 비정상({qty})")
        if not reason or len(reason) < 5:
            issues.append("주문 사유(reason) 누락/불충분")

        price = float(price_map.get(ticker) or 0.0)
        notional = price * max(qty, 0)
        ceiling = int(config.MAX_ORDER_QTY or 0) or int(config.HARD_MAX_ORDER_QTY)
        if side == "buy" and ceiling > 0 and qty > ceiling:
            issues.append(f"주문 수량({qty})이 1회 한도({ceiling}주) 초과")
        if side == "buy":
            if price <= 0:
                issues.append("현재가 조회 실패 → 사이즈/한도 검증 불가, 보수적 반려")
            if not bp_ok:
                issues.append("계좌 잔고 조회 실패 → 보수적 반려(매수 보류)")
            else:
                if pnl_ratio <= -abs(config.CONSERVATIVE_MDD):
                    issues.append(
                        f"계좌 평가손익 {pnl_ratio * 100:.1f}% — "
                        f"보수적 MDD(-{config.CONSERVATIVE_MDD * 100:.0f}%) 초과"
                    )
                if total > 0 and notional > total * config.CONSERVATIVE_STOCK_RATIO:
                    issues.append(
                        f"단일 종목 비중 {notional / total * 100:.1f}% — "
                        f"한도 {config.CONSERVATIVE_STOCK_RATIO * 100:.0f}% 초과"
                    )
                if cash > 0 and notional * config.MIN_CASH_BUFFER > cash:
                    issues.append(
                        f"예수금 부족: 필요 {notional * config.MIN_CASH_BUFFER:,.0f}원 > 보유 {cash:,.0f}원"
                    )
                if cash > 0 and spent + notional > cash * config.MAX_CYCLE_BUDGET_RATIO:
                    issues.append(f"사이클 누적 매수예산({cash * config.MAX_CYCLE_BUDGET_RATIO:,.0f}원) 초과")

        status = "REJECTED" if issues else "APPROVED"
        if status == "APPROVED" and side == "buy":
            spent += notional
        results.append({
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "price": price,
            "notional": notional,
            "status": status,
            "issues": issues,
            "entry_mode": raw.get("entry_mode"),
            "entry_limit": raw.get("entry_limit"),
        })

    approved = [x for x in results if x["status"] == "APPROVED"]
    lines = [
        f"주문 {len(results)}건 중 승인 {len(approved)}건",
        f"잔고: {'정상' if bp_ok else '조회실패'} | 예수금 {cash:,.0f}원 | 총평가 {total:,.0f}원 | 평가손익 {pnl_ratio * 100:.1f}% | 사이클 매수예산 사용 {spent:,.0f}원",
    ]
    for item in results:
        tag = "승인" if item["status"] == "APPROVED" else "반려"
        detail = "; ".join(item["issues"]) if item["issues"] else "전 항목 통과"
        px = f" @ {item['price']:,.0f}원 (≈{item['notional']:,.0f}원)" if item["price"] else ""
        lines.append(f"{tag} {item['ticker']} {item['side']} x{item['qty']}{px} — {detail}")
    return {"approved": bool(approved), "results": results, "report": "\n".join(lines)}
