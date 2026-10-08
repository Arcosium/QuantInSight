from __future__ import annotations

from typing import Any

from . import config
from .models import BuyingPower, Holding, OrderDraft
from .sessions import is_kr_code, is_mock_kr_tradable


SELL_HOLD_WORDS = {"보유", "유지", "hold", "keep", "유보", "관망"}
SELL_ALL_WORDS = {"전량", "전부", "모두", "all", "full", "100%", "청산"}
SELL_HALF_WORDS = {"절반", "반", "1/2", "half", "50%"}


def _sell_qty_from_directive(qty: int, directive: str) -> tuple[int, str]:
    text = str(directive or "").strip().lower()
    if not text or text in SELL_HOLD_WORDS:
        return 0, ""
    if text in SELL_ALL_WORDS:
        return qty, "사후관리실장 매도 판단 — 전량"
    if text in SELL_HALF_WORDS:
        return max(1, qty // 2), "사후관리실장 매도 판단 — 절반"
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        n = max(1, min(int(digits), qty))
        return n, f"사후관리실장 매도 판단 — {n}주"
    return qty, f"사후관리실장 매도 판단 — {directive}"


def assemble_sell_orders(
    holdings: list[Holding],
    *,
    sell_directives: dict[str, str] | None = None,
    total_eval: float,
) -> list[OrderDraft]:
    sell_directives = sell_directives or {}
    per_stock_cap = total_eval * config.CONSERVATIVE_STOCK_RATIO if total_eval > 0 else 0.0
    orders: list[OrderDraft] = []
    for h in holdings:
        if not is_kr_code(h.code) or h.qty <= 0:
            continue
        directive = sell_directives.get(h.code) or sell_directives.get(h.code.upper())
        sell_qty, reason = _sell_qty_from_directive(h.qty, directive or "")
        if not reason and config.ENABLE_SELL_REBALANCE:
            pnl = h.pnl_pct
            if pnl >= config.TAKE_PROFIT_PCT:
                sell_qty = h.qty
                reason = f"자동 익절 — 평가손익 {pnl:+.1f}% >= +{config.TAKE_PROFIT_PCT:.0f}%"
            elif pnl <= -abs(config.STOP_LOSS_PCT):
                sell_qty = h.qty
                reason = f"자동 손절 — 평가손익 {pnl:+.1f}% <= -{config.STOP_LOSS_PCT:.0f}%"
            elif config.TRIM_OVER_RATIO and per_stock_cap > 0 and h.notional > per_stock_cap:
                over_value = h.notional - per_stock_cap
                sell_qty = max(1, min(h.qty, int(over_value // max(h.cur_price, 1.0))))
                reason = f"편중 축소 — 단일종목 한도 {config.CONSERVATIVE_STOCK_RATIO * 100:.0f}% 초과"
        if reason and sell_qty > 0:
            if h.sellable_qty is not None:
                sell_qty = max(0, min(sell_qty, int(h.sellable_qty)))
            if sell_qty > 0:
                orders.append(OrderDraft(ticker=h.code, side="sell", qty=sell_qty, reason=reason))
    return orders


def _affordable_buy_qty(price: float, *, per_order_budget: float, per_stock_cap: float, cycle_remaining: float) -> int:
    if price <= 0:
        return 0
    budget = min(x for x in (per_order_budget, per_stock_cap, cycle_remaining) if x >= 0)
    return int(budget // price)


def assemble_orders(
    *,
    target_codes: list[str],
    holdings: list[Holding],
    buying_power: BuyingPower,
    price_map: dict[str, float],
    session: str,
    sell_directives: dict[str, str] | None = None,
    market_open: bool = False,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Build orders using QuantInSight's KR mock-account policy.

    Sells are risk-reducing and sorted before buys. Buys are KR only, skip held
    names, size from actual cash/total evaluation, and respect single-name and
    cycle caps. Mock accounts trade only during KR_TRADING.
    """
    notes: list[str] = []
    if not is_mock_kr_tradable(session):
        return {"orders": [], "sizing_notes": [f"현재 세션 {session}: 모의 국장 계정은 KRX 정규장만 매매"]}, price_map, buying_power.__dict__

    cash = float(buying_power.cash or 0.0)
    total = float(buying_power.total_eval or 0.0) or cash
    held_codes = {h.code for h in holdings}
    orders = assemble_sell_orders(holdings, sell_directives=sell_directives, total_eval=total)

    target_per_order = total * config.PER_ORDER_BUDGET_RATIO if total > 0 else 0.0
    if market_open:
        target_per_order = total * min(1.0, config.PER_ORDER_BUDGET_RATIO * 5.0)
    per_order_budget = min(target_per_order, cash) if cash > 0 else 0.0
    per_stock_cap = total * config.CONSERVATIVE_STOCK_RATIO if total > 0 else 0.0
    cycle_budget = cash * config.MAX_CYCLE_BUDGET_RATIO if cash > 0 else 0.0
    buy_names = [str(c).strip() for c in target_codes if is_kr_code(c) and str(c).strip() not in held_codes]
    per_name_budget = cycle_budget / max(1, len(buy_names)) if cycle_budget > 0 else per_order_budget
    spent = 0.0

    for code in buy_names[:8]:
        price = float(price_map.get(code) or 0.0)
        if price <= 0:
            notes.append(f"{code}: 현재가 없음 → 신규 매수 제외")
            continue
        qty = _affordable_buy_qty(
            price,
            per_order_budget=min(per_order_budget, per_name_budget),
            per_stock_cap=per_stock_cap if per_stock_cap > 0 else float("inf"),
            cycle_remaining=max(0.0, cycle_budget - spent),
        )
        ceiling = int(config.MAX_ORDER_QTY or 0) or int(config.HARD_MAX_ORDER_QTY)
        if ceiling > 0:
            qty = min(qty, ceiling)
        if qty < 1:
            if price <= cash and price <= max(0.0, cycle_budget - spent):
                qty = 1
                notes.append(f"{code}: 1주 {price:,.0f}원 — 예수금·사이클 잔여예산 이내 → 1주 매수")
            else:
                notes.append(f"{code}: 1주 {price:,.0f}원 — 예산 부족 → 제외")
                continue
        spent += qty * price
        orders.append(OrderDraft(
            ticker=code,
            side="buy",
            qty=qty,
            reason=(
                f"주식운용실장 지정 · {qty}주(≈{qty * price:,.0f}원, "
                f"총평가 {config.PER_ORDER_BUDGET_RATIO * 100:.0f}%·종목 "
                f"{config.CONSERVATIVE_STOCK_RATIO * 100:.0f}% 한도 내)"
            ),
        ))

    orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
    capped = build_exec_list([o.to_dict() for o in orders], config.MAX_TRADES_PER_CYCLE)
    return {"orders": capped, "sizing_notes": notes}, price_map, buying_power.__dict__


def build_exec_list(approved_orders: list[dict[str, Any]], max_trades: int) -> list[dict[str, Any]]:
    """QuantInSight execution cap: sells first, then buys up to cycle cap."""
    sells = [o for o in approved_orders if (o.get("side") or "buy") == "sell"]
    buys = [o for o in approved_orders if (o.get("side") or "buy") != "sell"]
    cap = max(0, int(max_trades or 0))
    return (sells + buys)[:cap]
