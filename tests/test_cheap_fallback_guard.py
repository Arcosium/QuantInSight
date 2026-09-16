"""Final committee choices cannot be replaced by unreviewed cheap names."""
import asyncio

import pytest

import main_swarm
from main_swarm import ArquantOrchestrator


class _StubRuntime:
    def __init__(self, params):
        self.params = params

    def get(self, key, uid=None, default=None):
        return self.params.get(key, default)


class _StubBroker:
    def __init__(self, prices):
        self.prices = prices  # {code: price}

    async def kr_account_snapshot(self):
        return {"buying_power": {"cash": 10_000_000.0, "total_eval": 10_000_000.0}, "holdings": []}

    async def kr_last_price(self, code):
        return float(self.prices.get(str(code), 0.0))

    async def us_last_price(self, tk):
        return float(self.prices.get(str(tk).upper(), 0.0))

    async def kr_psbl_order(self, code, price):
        return {"ok": False}  # 권위 조회 미가용 → clamp 안 함(현행 유지)


_PARAMS = {
    "PER_ORDER_BUDGET_RATIO": 0.2, "CONSERVATIVE_STOCK_RATIO": 0.25,
    "PER_ORDER_BUDGET_OVERSHOOT": 1.3, "MAX_ORDER_QTY": 0, "MAX_TRADES_PER_CYCLE": 2,
    "ENABLE_SELL_REBALANCE": True, "TAKE_PROFIT_PCT": 12.0, "STOP_LOSS_PCT": 5.0,
    "TRIM_OVER_RATIO": True, "ENABLE_CHEAP_FALLBACK": True, "ALLOW_US_STOCKS": True,
    "ALLOW_DERIVATIVES": False, "MAX_CYCLE_BUDGET_RATIO": 0.1,
}


def _orch(broker):
    o = object.__new__(ArquantOrchestrator)
    o.broker = broker
    o.uid = 1
    o.equity_path = "data/1/equity_curve.json"
    return o


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime(_PARAMS))
    monkeypatch.setattr(main_swarm, "get_current_session", lambda: "KR_TRADING")
    monkeypatch.setattr(main_swarm, "is_market_session_now", lambda *a, **k: False)  # equity 기록 경로 우회


def test_no_buy_when_trader_picks_nothing():
    # target 비어있음(의도적 매수 보류) + 저가 후보 존재 → 폴백이 발동하면 안 된다.
    b = _StubBroker({"000660": 200_000.0})
    obj, _px, _bp = asyncio.run(_orch(b)._build_orders(
        target_codes=[], candidate_codes=["000660"], quant_report="", news_report="",
        holdings=[], sell_directives={}))
    buys = [o for o in obj["orders"] if o.get("side") == "buy"]
    assert buys == [], "트레이더가 최종종목 없음으로 보류했으면 후보 최저가를 무단 매수하면 안 된다"



def test_cash_is_retained_when_designated_target_unaffordable():
    # Even a stale override cannot substitute an unreviewed cheap stock.
    b = _StubBroker({"005930": 15_000_000.0, "000660": 200_000.0})
    obj, _px, _bp = asyncio.run(_orch(b)._build_orders(
        target_codes=["005930"], candidate_codes=["000660"], quant_report="", news_report="",
        holdings=[], sell_directives={}))
    buys = [o for o in obj["orders"] if o.get("side") == "buy"]
    assert buys == []
    assert any("현금을 유지" in n for n in obj["sizing_notes"])


def test_two_targets_both_bought_with_split_budget():
    # 사장 지시 2026-06-03: 두 종목 이상 결정되면 하나만 사지 말고 예산을 나눠 둘 다 매수해야 한다.
    # cash/total=1,000만, MAX_CYCLE_BUDGET_RATIO=0.1 → 사이클 예산 100만, 2종목 → 종목당 50만씩 분배.
    b = _StubBroker({"005930": 70_000.0, "000660": 100_000.0})
    obj, _px, _bp = asyncio.run(_orch(b)._build_orders(
        target_codes=["005930", "000660"], candidate_codes=["005930", "000660"],
        quant_report="", news_report="", holdings=[], sell_directives={}))
    buys = {o["ticker"]: o for o in obj["orders"] if o.get("side") == "buy"}
    assert set(buys) == {"005930", "000660"}, "두 종목이 모두 매수돼야 한다(하나만 X)"
    # 각 종목은 분배된 예산(≈50만) 안에서 사이징 — 한 종목이 예산을 독식하지 않는다.
    assert buys["005930"]["qty"] * 70_000 <= 500_000 + 1  # 50만/7만 ≈ 7주
    assert buys["000660"]["qty"] * 100_000 <= 500_000 + 1  # 50만/10만 = 5주
    assert buys["005930"]["qty"] >= 1 and buys["000660"]["qty"] >= 1
