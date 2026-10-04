"""사장 지시 2026-06-09 #1: 슬리브 매도를 사후관리실장이 *종합*한다(구 strip 폐지).

새 흐름: _cyc_stage_sleeves(매니저가 매크로+뉴스로 매도 제안) → _cyc_stage_finalize_sell
(사후관리실장이 주식 매도 + 슬리브 제안을 종합). 슬리브 ETF 는 사후관리실장의 *주식* 매도
평가 입력(holdings_str)에선 제외되지만, 별도 '매도 제안' 블록으로 보여 사후관리실장이 매도결정에
포함(종합)할 수 있다. 슬리브 OFF 면 기존대로 슬리브도 주식 트랙에 노출(동작 불변).
"""
import asyncio

import pytest

import main_swarm
import runtime


def _mk_orchestrator(post_view):
    sm = main_swarm.ArquantOrchestrator.__new__(main_swarm.ArquantOrchestrator)
    sm.uid = "test"
    captured = {"pm_prompt": None}

    class _Agent:
        def __init__(self, view):
            self._view = view

        async def think(self, prompt):
            captured["pm_prompt"] = prompt
            return self._view

    class _Orch:
        async def think(self, prompt):
            return "최종종목: 없음"

    sm.post_manager = _Agent(post_view)
    sm.orchestrator = _Orch()

    class _Log:
        def log(self, *a, **k):
            pass
    sm.cycle_log = _Log()

    async def _noop_emit(*a, **k):
        return None
    sm._emit = _noop_emit
    return sm, captured


def _mk_cyc(sleeve_sell_proposals=None):
    class _Cyc:
        pass
    cyc = _Cyc()
    cyc.session = "KR_TRADING"
    cyc.market_open = True
    cyc._sell_only = True
    cyc.standing_directive_block = ""
    cyc.candidate_codes = []
    cyc._cand_line = ""
    cyc.macro_report = ""
    cyc.quant_report = ""
    cyc.news_report = ""
    cyc.holdings = [
        {"code": "153130", "name": "KODEX 단기채권", "qty": 100, "cur_price": 100_000, "pnl_pct": 12.0},
        {"code": "005930", "name": "삼성전자", "qty": 10, "cur_price": 70_000, "pnl_pct": 12.0},
    ]
    cyc.holdings_str = "; ".join(f"{h['name']}({h['code']}) {h['qty']}주" for h in cyc.holdings)
    cyc._budget_hint = ""
    cyc.index_facts = ""
    cyc.stock_holdings = None
    cyc.thesis_reminders = {}
    cyc.sleeve_sell_proposals = sleeve_sell_proposals or {}
    return cyc


@pytest.fixture(autouse=True)
def _stub_holding_period(monkeypatch):
    monkeypatch.setattr(main_swarm.cycle_store, "get_holding_period", lambda code, uid=0: None)


def _run(sm, cyc):
    asyncio.run(sm._cyc_stage_finalize_sell(cyc))


def _runtime_get_factory(sleeve_on):
    def _get(key, default=None, uid=None):
        if key in ("ENABLE_BOND_ETF", "ENABLE_COMMODITY_ETF"):
            return sleeve_on
        if key == "MAX_TRADES_PER_CYCLE":
            return 2
        if key == "ALLOW_DAY_TRADING":
            return True
        if key == "MIN_HOLDING_DAYS_FOR_SELL":
            return 0.0
        if key in ("MIN_QUANT_SCORE", "MAX_BUY_NAMES"):
            return 0
        return default
    return _get


def test_sleeve_on_excludes_from_stock_track_but_synthesizes_sells(monkeypatch):
    """슬리브 ON: 사후관리실장의 *주식* 입력엔 채권 없음. 단 매도 제안 블록을 보고 매도결정에
    채권 코드를 포함하면 그 결정이 *종합*되어 sell_directives 에 남는다."""
    monkeypatch.setattr(runtime, "get", _runtime_get_factory(sleeve_on=True))
    sm, captured = _mk_orchestrator("매도결정: 005930=전량, 153130=전량")
    cyc = _mk_cyc(sleeve_sell_proposals={"bond": {"153130": "전량"}})
    _run(sm, cyc)

    # 1) 사후관리실장 *주식* 매도 평가 입력(stock_holdings)은 채권 제외
    assert {h["code"] for h in cyc.stock_holdings} == {"005930"}
    # 2) 프롬프트에 슬리브 매도 제안 블록이 들어가 사후관리실장이 종합하게 한다
    assert "매도 제안" in (captured["pm_prompt"] or "")
    # 3) 슬리브 매도결정이 *종합*되어 sell_directives 에 남는다(구 strip 폐지)
    assert cyc.sell_directives.get("153130") == "전량"
    assert cyc.sell_directives.get("005930") == "전량"
    # 4) cyc.holdings 는 전체(슬리브 포함) 보존
    assert {h["code"] for h in cyc.holdings} == {"153130", "005930"}


def test_sleeve_proposal_merged_even_if_post_manager_omits(monkeypatch):
    """사후관리실장이 슬리브 코드를 안 다루면, 매니저 제안이 그대로 반영(누락 방지)."""
    monkeypatch.setattr(runtime, "get", _runtime_get_factory(sleeve_on=True))
    sm, captured = _mk_orchestrator("매도결정: 005930=보유")
    cyc = _mk_cyc(sleeve_sell_proposals={"bond": {"153130": "절반"}})
    _run(sm, cyc)
    assert cyc.sell_directives.get("153130") == "절반"  # 제안 유지


def test_sleeve_off_keeps_existing_behavior(monkeypatch):
    monkeypatch.setattr(runtime, "get", _runtime_get_factory(sleeve_on=False))
    sm, captured = _mk_orchestrator("매도결정: 005930=전량, 153130=전량")
    cyc = _mk_cyc()
    _run(sm, cyc)

    # OFF: 기존대로 채권도 사후관리실장에 노출되고 매도결정에 반영(동작 불변)
    assert "153130" in (captured["pm_prompt"] or "")
    assert cyc.sell_directives.get("153130") == "전량"
    assert cyc.sell_directives.get("005930") == "전량"
    assert getattr(cyc, "stock_holdings", None) is None
