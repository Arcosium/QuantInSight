"""Deterministic, bounded order-book experiments. No broker or live-order imports.

The replenishment model restores displayed quantity at the original price levels.
It is a conditional toy model, not a counterfactual replay of historical prices.
"""
from copy import deepcopy
import math


def validate_book(book):
    result = {}
    for name, reverse in (("bids", True), ("asks", False)):
        levels = book.get(name, [])
        if not levels or len(levels) > 1000:
            raise ValueError("호가 자료는 매수·매도 각각 1~1000단계가 필요합니다.")
        seen = set()
        parsed = []
        for price, quantity in levels:
            p, q = float(price), float(quantity)
            if not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0 or p in seen:
                raise ValueError("호가 가격·잔량은 유한한 양수이며 가격은 중복될 수 없습니다.")
            seen.add(p)
            parsed.append([p, q])
        result[name] = sorted(parsed, reverse=reverse)
    if result["bids"][0][0] >= result["asks"][0][0]:
        raise ValueError("매수·매도 호가가 교차한 자료입니다.")
    return result


def quote(book):
    bids = [p for p, q in book["bids"] if q > 1e-10]
    asks = [p for p, q in book["asks"] if q > 1e-10]
    if not bids or not asks:
        return {"mid": None, "spread": None}
    return {"mid": (max(bids) + min(asks)) / 2, "spread": min(asks)-max(bids)}


def consume(book, side, quantity):
    """Mutate only the passed book; never fill beyond the displayed depth."""
    if side not in ("buy", "sell") or not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("주문 방향과 양수 수량을 확인하세요.")
    levels = book["asks" if side == "buy" else "bids"]
    remaining = quantity
    fills = []
    for level in levels:
        taken = min(max(0.0, level[1]), remaining)
        if taken > 1e-10:
            fills.append({"price": level[0], "quantity": taken})
            level[1] -= taken
            remaining -= taken
        if remaining <= 1e-10:
            remaining = 0.0
            break
    filled = quantity-remaining
    notional = sum(row["price"]*row["quantity"] for row in fills)
    return {"filled": filled, "unfilled": remaining, "notional": notional,
            "vwap": notional/filled if filled else None, "fills": fills}


def replenish(book, initial, elapsed, half_time):
    if half_time is None or elapsed <= 0:
        return
    fraction = -math.expm1(-math.log(2)*elapsed/half_time)
    for name in ("bids", "asks"):
        for current, original in zip(book[name], initial[name]):
            current[1] += max(0.0, original[1]-current[1])*fraction


def _scenario(initial, side, quantity, slices, horizon, half_time, fee_bps):
    book = deepcopy(initial)
    before = quote(initial)
    sign = 1 if side == "buy" else -1
    levels_key = "asks" if side == "buy" else "bids"
    best = initial[levels_key][0][0]
    schedule = [horizon*i/(slices-1) for i in range(slices)] if slices > 1 else [0.0]
    grid = sorted(set([round(horizon*i/100, 9) for i in range(101)] + schedule))
    executions, path = [], []
    elapsed, next_slice = 0.0, 0
    for at in grid:
        replenish(book, initial, at-elapsed, half_time)
        elapsed = at
        while next_slice < len(schedule) and schedule[next_slice] <= at+1e-10:
            fill = consume(book, side, quantity/slices)
            fill["at_seconds"] = at
            executions.append(fill)
            next_slice += 1
        after = quote(book)
        path.append({"seconds": at,
                     "impact_bps": sign*(after["mid"]-before["mid"])/before["mid"]*1e4 if after["mid"] is not None else None,
                     "remaining_depth": sum(q for _, q in book[levels_key]),
                     "remaining_top5_depth": sum(q for _, q in book[levels_key][:5]),
                     "spread_bps": after["spread"]/before["mid"]*1e4 if after["spread"] is not None else None})
    filled = sum(e["filled"] for e in executions)
    notional = sum(e["notional"] for e in executions)
    vwap = notional/filled if filled else None
    cost = sign*(vwap-before["mid"])/before["mid"]*1e4 if vwap else None
    walk = sign*(vwap-best)/before["mid"]*1e4 if vwap else None
    # Fees scale with executed notional, denominated by arrival-mid notional.
    fees = fee_bps*vwap/before["mid"] if vwap else None
    return {"slices": slices, "requested": quantity, "filled": filled,
            "unfilled": max(0.0, quantity-filled), "fill_fraction": filled/quantity,
            "complete": math.isclose(filled, quantity, rel_tol=1e-9, abs_tol=1e-9),
            "vwap": vwap, "cost_bps": cost, "book_walk_bps": walk,
            "fee_bps_effective": fees, "total_cost_bps": cost+fees if cost is not None else None,
            "initial_impact_bps": path[0]["impact_bps"],
            "end_impact_bps": path[-1]["impact_bps"], "path": path,
            "executions": executions, "book_after_first": _first_book(initial, side, quantity/slices),
            "book_at_end": book}


def _first_book(initial, side, quantity):
    book = deepcopy(initial)
    consume(book, side, quantity)
    return book


def simulate(book, *, side="buy", quantity=1.0, slices=5, horizon=5.0,
             half_time=None, fee_bps=0.0):
    initial = validate_book(book)
    if side not in ("buy", "sell"):
        raise ValueError("매수 또는 매도를 선택하세요.")
    for value in (quantity, horizon, fee_bps):
        if not math.isfinite(value):
            raise ValueError("입력값은 유한한 숫자여야 합니다.")
    if not 1e-8 <= quantity <= 1e9 or not 0.1 <= horizon <= 30 or not 0 <= fee_bps <= 100:
        raise ValueError("수량·기간·수수료의 허용 범위를 확인하세요.")
    if type(slices) is not int or not 2 <= slices <= 20:
        raise ValueError("분할 횟수는 2~20 사이 정수입니다.")
    if half_time is not None and (not math.isfinite(half_time) or not 0.01 <= half_time <= 300):
        raise ValueError("잔량 부족분 반감기는 0.01~300초입니다.")
    q = quote(initial)
    single = _scenario(initial, side, quantity, 1, horizon, half_time, fee_bps)
    split = _scenario(initial, side, quantity, slices, horizon, half_time, fee_bps)
    comparable = (single["complete"] and split["complete"]
                  and single["total_cost_bps"] is not None and split["total_cost_bps"] is not None)
    return {"model": "fixed-price displayed-depth replenishment",
            "side": side, "initial": initial, "arrival_mid": q["mid"],
            "horizon_seconds": horizon, "half_time_seconds": half_time, "fee_bps": fee_bps,
            "no_action": {"mid": q["mid"], "impact_bps": 0.0, "filled": 0.0},
            "single": single, "split": split, "cost_comparable": comparable,
            "split_saving_bps": single["total_cost_bps"]-split["total_cost_bps"] if comparable else None,
            "assumptions": ["원래 가격대의 표시 잔량만 회복", "외부 가격 추세·타 참여자 반응·대기열·지연 미반영",
                            "관측 호가 밖 잔량은 체결하지 않음", "분할 주문 미체결분은 다음 회차에 재주문하지 않음",
                            "모든 비용은 실험 시작 mid price를 기준으로 계산"]}
