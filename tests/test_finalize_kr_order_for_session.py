"""주문을 세션 거래소로 데코레이트 + 시간외 지정가 산정 + 시세결손 보류."""
import asyncio
import main_swarm
from infra.kis_broker import OrderDraft

class _StubRuntime:
    def __init__(self, params): self._p = params
    def get(self, k, default=None, uid=None): return self._p.get(k, default)  # uid 무시(전역 스텁)

class _StubBroker:
    def __init__(self, px): self._px = px; self.calls = []; self.is_mock = False; self._nxt = None
    async def kr_last_price(self, code, market="J"):
        self.calls.append(market); return self._px
    def nxt_supported(self): return self._nxt


class _MarketPriceBroker(_StubBroker):
    def __init__(self, prices):
        super().__init__(0)
        self.prices = prices

    async def kr_last_price(self, code, market="J"):
        self.calls.append(market)
        return self.prices.get(market, 0)

def _orch():
    o = main_swarm.ArquantOrchestrator.__new__(main_swarm.ArquantOrchestrator)
    o.uid = None   # runtime.get(uid=self.uid) 가 참조 — __new__ 객체엔 미설정이므로 명시
    return o

def test_regular_session_sets_krx_no_pricing(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5}))
    o = _orch(); o.broker = _StubBroker(10000)
    od = OrderDraft(ticker="005930", side="buy", qty=1, price_type="market", market="KR", approved=True)
    res, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_TRADING"))
    assert res.exchange == "KRX"
    assert skip is None
    assert o.broker.calls == []

def test_extended_session_sets_nxt_and_limit(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5}))
    o = _orch(); o.broker = _StubBroker(10000)
    od = OrderDraft(ticker="005930", side="buy", qty=1, price_type="market", market="KR", approved=True)
    res, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_AFTER_MARKET"))
    assert res.exchange == "NXT"
    assert res.price_type.value == "limit"
    assert res.limit_price == 10050   # NX·정규가 동일 → 슬리피지 밴드만 적용(캡 불발동)
    assert skip is None
    # NX(주문가) + J(정규 기준가, 프리미엄 캡용) 둘 다 조회한다.
    assert o.broker.calls == ["NX", "J"]


def test_extended_session_caps_nxt_premium(monkeypatch):
    """NXT가 정규가보다 크게 프리미엄(+4.4%)일 때 매수 지정가를 정규가 기준 캡으로 제한."""
    from infra.kis_broker import round_to_tick
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime(
        {"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5, "EXT_HOURS_MAX_PREMIUM_PCT": 1.5}))
    o = _orch(); o.broker = _MarketPriceBroker({"NX": 27776, "J": 26600})
    od = OrderDraft(ticker="003490", side="buy", qty=1, market="KR", approved=True)
    res, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_PRE_MARKET"))
    assert skip is None
    assert res.limit_price == round_to_tick(26600 * 1.015)   # NX 프리미엄 추종 안 함
    assert res.limit_price < round_to_tick(27776 * 1.005)

def test_extended_session_no_price_holds(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5}))
    o = _orch(); o.broker = _StubBroker(0)
    od = OrderDraft(ticker="005930", side="buy", qty=1, market="KR", approved=True)
    res, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_PRE_MARKET"))
    assert skip is not None and "시세" in skip


def test_extended_session_does_not_use_krx_price_for_nxt_order(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5}))
    o = _orch(); o.broker = _MarketPriceBroker({"NX": 0, "J": 113300})
    od = OrderDraft(ticker="153130", side="sell", qty=7, market="KR", approved=True)
    _, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_AFTER_MARKET"))
    assert skip is not None and "NXT" in skip
    assert o.broker.calls == ["NX"]

def test_us_order_untouched(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({"EXT_HOURS_LIMIT_SLIPPAGE_PCT": 0.5}))
    o = _orch(); o.broker = _StubBroker(10000)
    od = OrderDraft(ticker="AAPL", side="buy", qty=1, market="US", approved=True)
    res, skip = asyncio.run(o._finalize_kr_order_for_session(od, "KR_AFTER_MARKET"))
    assert res is od            # US 주문은 그대로 반환 — 거래소/가격 손대지 않음
    assert skip is None
    assert o.broker.calls == []

def test_extended_hours_hard_blocked_on_mock(monkeypatch):
    # 사장 지시 2026-06-08: 모의 계정은 NXT 시간외 아예 비활성(능력감지보다 앞선 하드 게이트)
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime({}))
    o = _orch(); o.broker = _StubBroker(10000); o.broker.is_mock = True
    reason = o._extended_hours_blocked("KR_AFTER_MARKET")
    assert reason and "모의" in reason

def test_extended_hours_not_blocked_live_default(monkeypatch):
    monkeypatch.setattr(main_swarm, "runtime", _StubRuntime(
        {"ENABLE_NXT_EXTENDED_HOURS": True, "ENABLE_NXT_AFTER_MARKET": True}))
    o = _orch(); o.broker = _StubBroker(10000)  # is_mock=False, nxt_supported=None
    assert o._extended_hours_blocked("KR_AFTER_MARKET") is None
