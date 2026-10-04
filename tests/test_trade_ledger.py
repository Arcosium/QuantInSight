"""실거래 원장(infra/trade_ledger) — KIS 집계 TR 비의존 자산평가 (사장 지시 2026-06-11).

배경: KIS 통합총자산 TR 자기불일치·USD 결제 과도기·해외평가 증발로 수익률 KPI 가
환각(-43% 표시)을 일으켰다. 원장은 ① 1회 시드(KIS 보유/예수금) ② 이후 우리 체결만으로
현금·포지션 진화 ③ 자체 시세 M2M 으로 평가한다.
"""
import asyncio
import json

import pytest

from infra import trade_ledger as tl


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path, monkeypatch):
    """라이브 data/ 오염 방지 — 모든 테스트는 tmp 디렉토리 원장만 만진다."""
    monkeypatch.setattr(tl, "_DATA_DIR", tmp_path)
    return tmp_path


class _Broker:
    def __init__(self, *, is_mock=False, us_prices=None, pk=None, fills=None):
        self.is_mock = is_mock
        self._us_prices = us_prices or {}
        self._pk = pk or {"ok": False}
        self._fills = fills or []

    async def us_last_price(self, tk):
        return self._us_prices.get(tk, 0.0)

    async def _overseas_present_krw(self):
        return self._pk

    async def overseas_fills(self, start_ymd, end_ymd):
        return list(self._fills)


def _snap(cash=1_000_000.0, holdings=None, ok=True):
    return {"buying_power": {"cash": cash, "ok": ok}, "holdings": holdings or []}


# ── 시드 ──────────────────────────────────────────────────────────────────────

def test_seed_live_kr_and_usd_cash():
    """실전: KR 보유 평단 신뢰 + 외화예수금(CTRP6504R)을 USD 현금으로 시드."""
    br = _Broker(pk={"ok": True, "exrt": 1500.0, "deposit_krw": 1_500_000.0})
    snap = _snap(cash=4_000_000, holdings=[
        {"code": "153130", "qty": 3, "avg_price": 113301.0, "cur_price": 113300.0},
    ])
    led = asyncio.run(tl.seed(1, br, snap))
    assert led is not None
    assert led["cash_krw"] == 4_000_000
    assert led["cash_usd"] == pytest.approx(1000.0)  # 1.5M KRW / 1500
    assert led["positions"]["153130"]["qty"] == 3
    assert led["positions"]["153130"]["avg_cost"] == pytest.approx(113301.0)
    assert led["positions"]["153130"]["ccy"] == "KRW"


def test_seed_skips_on_balance_glitch():
    """bp.ok=False(잔고 글리치) 시점엔 시드하지 않는다 — 오염된 베이스라인 방지."""
    led = asyncio.run(tl.seed(1, _Broker(), _snap(ok=False)))
    assert led is None
    assert tl.load(1) is None


def test_seed_mock_us_uses_live_price_not_garbage():
    """모의: 해외 보유 평단/시세는 garbage → us_last_price 실시세로 근사(approx_basis)."""
    br = _Broker(is_mock=True, us_prices={"GLD": 310.0})
    snap = _snap(cash=90_000_000, holdings=[
        {"code": "GLD", "qty": 4, "avg_price": 99999.0, "cur_price": 0.5, "ccy": "USD"},
    ])
    led = asyncio.run(tl.seed(2, br, snap))
    pos = led["positions"]["GLD"]
    assert pos["avg_cost"] == pytest.approx(310.0)
    assert pos["last_price"] == pytest.approx(310.0)
    assert pos["approx_basis"] is True
    assert led["cash_usd"] == 0.0  # 모의 외화예수금은 신뢰 불가 → 0 시드


# ── 미결제 USD (T+2) — 2026-06-11 라이브 디버깅 사례 ─────────────────────────────

_LIVE_FILLS = [  # uid1 실사례 축약: 06-09 순매수(통합증거금→KRW 차감), 06-10 순매도(미결제 USD)
    {"date": "20260608", "ticker": "SHY", "side": "buy", "qty": 16, "price": 81.905, "amount": 1310.48, "ccy": "USD"},
    {"date": "20260609", "ticker": "DAL", "side": "buy", "qty": 3, "price": 80.10, "amount": 240.30, "ccy": "USD"},
    {"date": "20260609", "ticker": "GLW", "side": "sell", "qty": 1, "price": 169.89, "amount": 169.89, "ccy": "USD"},
    {"date": "20260610", "ticker": "SHY", "side": "sell", "qty": 16, "price": 81.975, "amount": 1311.60, "ccy": "USD"},
    {"date": "20260610", "ticker": "USO", "side": "buy", "qty": 1, "price": 135.4699, "amount": 135.4699, "ccy": "USD"},
]


def test_pending_usd_counts_only_unsettled_positive_nets():
    """06-10 순매도(+1176.13)만 미결제 USD — 06-09 순매수(음수)는 KRW(통합증거금) 몫,
    06-08 매수는 T+2 경과(결제완료)라 제외."""
    import datetime as _dt
    br = _Broker(fills=_LIVE_FILLS)
    pend = asyncio.run(tl.pending_usd_from_fills(br, today_us=_dt.date(2026, 6, 10)))
    assert pend == pytest.approx(1311.60 - 135.4699)


def test_pending_usd_settles_after_two_bdays():
    """이틀(영업일) 지나면 결제 완료 → 미결제 0 (그때는 외화예수금 TR 이 잡는다)."""
    import datetime as _dt
    br = _Broker(fills=_LIVE_FILLS)
    pend = asyncio.run(tl.pending_usd_from_fills(br, today_us=_dt.date(2026, 6, 12)))
    assert pend == 0.0


def test_seed_adds_pending_usd_to_cash():
    br = _Broker(pk={"ok": True, "exrt": 1500.0, "deposit_krw": 0.0}, fills=[
        {"date": "20991231", "ticker": "XOM", "side": "sell", "qty": 1, "price": 100.0,
         "amount": 100.0, "ccy": "USD"},   # 항상 미결제(미래 날짜)
    ])
    led = asyncio.run(tl.seed(1, br, _snap(cash=1_000_000)))
    assert led["cash_usd"] == pytest.approx(100.0)
    assert led["seed_pending_usd"] == pytest.approx(100.0)


# ── 체결 반영 ─────────────────────────────────────────────────────────────────

def _seed_basic(uid=1):
    br = _Broker(pk={"ok": True, "exrt": 1500.0, "deposit_krw": 0.0})
    snap = _snap(cash=1_000_000, holdings=[
        {"code": "AAPL", "qty": 10, "avg_price": 100.0, "cur_price": 100.0, "ccy": "USD"},
    ])
    return asyncio.run(tl.seed(uid, br, snap))


def test_apply_fill_before_seed_is_noop():
    """시드 전 체결은 무시 — 시드 스냅샷이 이미 그 체결을 반영하므로 이중계상 방지."""
    assert tl.apply_fill(7, ticker="005930", side="buy", qty=1, price=60000, ccy="KRW") is False


def test_apply_fill_us_buy_with_fee():
    """US 매수: 현금 차감 = 대금 + 0.3% 수수료, 평단 블렌딩."""
    _seed_basic()
    assert tl.apply_fill(1, ticker="AAPL", side="buy", qty=10, price=110.0, ccy="USD")
    led = tl.load(1)
    pos = led["positions"]["AAPL"]
    assert pos["qty"] == 20
    assert pos["avg_cost"] == pytest.approx(105.0)        # (100×10 + 110×10)/20
    assert led["cash_usd"] == pytest.approx(-(110 * 10) - 0.003 * 110 * 10)


def test_apply_fill_us_sell_removes_position_and_credits_cash():
    """US 전량 매도: 현금 += 대금 − 0.3% 수수료, 포지션 제거."""
    _seed_basic()
    assert tl.apply_fill(1, ticker="AAPL", side="sell", qty=10, price=120.0, ccy="USD")
    led = tl.load(1)
    assert "AAPL" not in led["positions"]
    assert led["cash_usd"] == pytest.approx(120 * 10 - 0.003 * 120 * 10)


def test_apply_fill_accepts_orderside_enum():
    """OrderSide(str,Enum)이 side 로 넘어와도 반영돼야 한다 — str(enum)='OrderSide.SELL' 이
    조용히 거부돼 uid2 132030 매도 72주가 원장 미반영된 라이브 사례(2026-06-11) 회귀 방지."""
    from infra.kis_broker import OrderSide
    _seed_basic()
    assert tl.apply_fill(1, ticker="AAPL", side=OrderSide.SELL, qty=10, price=120.0, ccy="USD")
    assert "AAPL" not in tl.load(1)["positions"]


def test_apply_fill_kr_includes_assumed_entry_cost():
    """Entry cash includes costs, applied only to new fills."""
    br = _Broker()
    snap = _snap(cash=1_000_000, holdings=[])
    asyncio.run(tl.seed(1, br, snap))
    tl.apply_fill(1, ticker="005930", side="buy", qty=2, price=60000, ccy="KRW")
    led = tl.load(1)
    assert led["cash_krw"] == pytest.approx(1_000_000 - 120_000*(1+tl.KR_BUY_COST_RATE))
    assert led["positions"]["005930"]["avg_cost"] == pytest.approx(60000)


def test_apply_fill_sell_price_fallback_to_last_price():
    """매도 체결가 미상(US 폴링 결손)이면 직전가로 근사 — 누락(절대 금지)보다 근사."""
    _seed_basic()
    assert tl.apply_fill(1, ticker="AAPL", side="sell", qty=5, price=None, ccy="USD")
    led = tl.load(1)
    assert led["positions"]["AAPL"]["qty"] == 5
    assert led["cash_usd"] > 0
    assert led["fills"][-1]["approx_price"] is True


def test_kr_partial_sale_attributes_entry_cost_without_double_charge():
    asyncio.run(tl.seed(1, _Broker(), _snap(cash=1_000_000, holdings=[])))
    tl.apply_fill(1, ticker="005930", side="buy", qty=10, price=1000)
    tl.apply_fill(1, ticker="005930", side="sell", qty=4, price=1100)
    led = tl.load(1)
    assert led["fills"][-1]["realized"] == pytest.approx(400-4400*tl.KR_SELL_COST_RATE-4000*tl.KR_BUY_COST_RATE)
    assert led["positions"]["005930"]["entry_fees"] == pytest.approx(6000*tl.KR_BUY_COST_RATE)
    tl.apply_fill(1, ticker="005930", side="sell", qty=6, price=1100)
    led = tl.load(1)
    assert sum(f.get("realized",0) for f in led["fills"]) == pytest.approx(1000-10000*tl.KR_BUY_COST_RATE-11000*tl.KR_SELL_COST_RATE)
    assert led["cash_krw"]-1_000_000 == pytest.approx(sum(f.get("realized",0) for f in led["fills"]))


def test_apply_fill_unknown_sell_is_degraded_not_applied():
    """원장에 없는 종목 매도(시드 전 매수분 등)는 현금 왜곡 없이 degraded 카운트만."""
    _seed_basic()
    before = tl.load(1)["cash_usd"]
    assert tl.apply_fill(1, ticker="TSLA", side="sell", qty=3, price=200.0, ccy="USD") is False
    led = tl.load(1)
    assert led["cash_usd"] == pytest.approx(before)
    assert led["degraded_fills"] == 1


# ── M2M 평가 ─────────────────────────────────────────────────────────────────

def test_mark_to_market_carries_forward_missing_price():
    """시세 결손 종목은 직전가 carry-forward — 곡선이 0 으로 튀지 않는다."""
    _seed_basic()
    m1 = tl.mark_to_market(1, price_lookup={"AAPL": 110.0}, fx=1500.0)
    assert m1["value_krw"] == pytest.approx(1_000_000 + 10 * 110 * 1500)
    # 다음 폴엔 AAPL 시세 결손 → 110 유지
    m2 = tl.mark_to_market(1, price_lookup={}, fx=1500.0)
    assert m2["value_krw"] == pytest.approx(m1["value_krw"])
    assert m2["stale"] == ["AAPL"]


def test_value_includes_usd_cash():
    """USD 예수금이 평가에 포함된다 — US 매도 후 '자산 증발' 환각의 직접 수정."""
    br = _Broker(pk={"ok": True, "exrt": 1500.0, "deposit_krw": 1_500_000.0})
    asyncio.run(tl.seed(1, br, _snap(cash=1_000_000)))
    m = tl.mark_to_market(1, price_lookup={}, fx=1500.0)
    assert m["value_krw"] == pytest.approx(1_000_000 + 1000 * 1500)


def test_ensure_value_seeds_then_values():
    """ensure_value: 원장 없으면 시드 후 즉시 평가값 반환 (폴러 원스톱)."""
    br = _Broker(pk={"ok": False})
    snap = _snap(cash=2_000_000, holdings=[
        {"code": "153130", "qty": 3, "avg_price": 113301.0, "cur_price": 113300.0},
    ])
    v = asyncio.run(tl.ensure_value(1, br, snap, fx=1500.0))
    assert v == pytest.approx(2_000_000 + 3 * 113300.0)
    assert tl.load(1) is not None


# ── reconcile ────────────────────────────────────────────────────────────────

def test_reconcile_detects_qty_drift():
    _seed_basic()
    diffs = tl.reconcile(1, [{"code": "AAPL", "qty": 7}])
    assert diffs and "AAPL" in diffs[0]


def test_reconcile_empty_holdings_skipped():
    """KIS 보유 스냅샷 일시 결손(빈 목록)은 비교하지 않는다 — 글리치 오탐 방지."""
    _seed_basic()
    assert tl.reconcile(1, []) == []


def test_repair_recent_partial_buy_after_restart(monkeypatch):
    """부분체결 잔여 폴링 task 가 재시작으로 유실돼도, 최근 cycle 기록+KIS 잔고로 원장을 보정한다.

    라이브 사례: 039830 204주 주문 중 즉시 139주만 확인 → 원장 139주. 재시작 후 KIS 잔고는
    204주로 반영됐지만 polling task 가 사라져 65주가 원장 누락됐다.
    """
    from infra import cycle_store

    asyncio.run(tl.seed(2, _Broker(), _snap(cash=10_000_000, holdings=[])))
    assert tl.apply_fill(2, ticker="039830", side="buy", qty=139,
                         price=18668.417, ccy="KRW", avg_cost=18668.417)
    monkeypatch.setattr(cycle_store, "list_cycles", lambda limit=30, uid=None: [{
        "orders_executed": json.dumps([{
            "ticker": "039830", "side": "buy", "qty": 139, "order_qty": 204,
            "accepted": True, "filled": True, "fill_price": 18668.417,
            "avg_cost": 18668.417,
        }], ensure_ascii=False)
    }])

    # 버그 D(2026-06-18): 누락매수 상향보정은 KIS 글리치-高 방어를 위해 '연속 확인' 후 적용한다
    # (LEDGER_REPAIR_CONFIRMATIONS=2). 1회차는 스트릭 적립만, 2회차에 보정.
    _kis = [{"code": "039830", "qty": 204, "avg_price": 18653.088}]
    assert tl.repair_from_recent_partial_orders(2, _kis) == []        # 1회차: 확인 적립
    assert tl.load(2)["positions"]["039830"]["qty"] == 139            # 아직 미보정

    repaired = tl.repair_from_recent_partial_orders(2, _kis)          # 2회차: 연속 확인 → 보정
    assert repaired == ["039830: 누락 매수 65주 원장 보정"]
    led = tl.load(2)
    assert led["positions"]["039830"]["qty"] == 204
    assert tl.reconcile(2, [{"code": "039830", "qty": 204}]) == []


def test_repair_recent_unconfirmed_sell_after_restart(monkeypatch):
    """접수만 됐던 매도 주문의 polling task 가 재시작으로 유실된 뒤 KIS 잔고가 0이면 원장 매도를 보정."""
    from infra import cycle_store

    asyncio.run(tl.seed(2, _Broker(), _snap(cash=10_000_000, holdings=[
        {"code": "036570", "qty": 77, "avg_price": 264603.896, "cur_price": 258000.0},
    ])))
    monkeypatch.setattr(cycle_store, "list_cycles", lambda limit=30, uid=None: [{
        "orders_executed": json.dumps([{
            "ticker": "036570", "side": "sell", "qty": 77, "order_qty": 77,
            "accepted": True, "filled": False, "avg_cost": 264603.896,
        }], ensure_ascii=False)
    }])

    repaired = tl.repair_from_recent_partial_orders(2, [])
    assert repaired == []

    repaired = tl.repair_from_recent_partial_orders(2, [{"code": "161890", "qty": 1}])

    assert repaired == ["036570: 누락 매도 77주 원장 보정"]
    assert "036570" not in tl.load(2)["positions"]


def test_restart_repair_does_not_double_book_active_partial_fill_poll(monkeypatch):
    from infra import cycle_store
    asyncio.run(tl.seed(2, _Broker(), _snap(cash=10_000_000, holdings=[
        {"code": "153130", "qty": 57, "avg_price": 100., "cur_price": 100.},
    ])))
    pending = [{"ticker": "153130", "side": "sell", "qty": 28, "order_qty": 28,
                "accepted": True, "filled": False, "fill_price": 100.}]
    monkeypatch.setattr(cycle_store, "list_cycles", lambda **kw: [{"orders_executed": json.dumps(pending)}])
    before = tl.load(2)
    with tl.pending_fill_poll(2, pending):
        assert tl.repair_from_recent_partial_orders(2, [{"code": "153130", "qty": 55}]) == []
        assert tl.prune_phantoms(2, [{"code": "153130", "qty": 55}], min_confirmations=1)["pruned"] == []
        assert tl.adopt_orphans(2, [{"code": "153130", "qty": 59, "avg_price": 100.}], min_confirmations=1)["adopted"] == []
        assert tl.load(2)["positions"]["153130"]["qty"] == 57
        assert tl.load(2)["cash_krw"] == before["cash_krw"]
        # The active poll later confirms the total; it must be applied once.
        assert tl.apply_fill(2, ticker="153130", side="sell", qty=28, price=100., note="poll_confirm")
    assert not tl.fill_poll_active(2, "153130")
    assert tl.load(2)["positions"]["153130"]["qty"] == 29


def test_fill_poll_ownership_is_account_scoped_and_released_on_cancellation():
    pending = [{"ticker": "153130"}]
    with pytest.raises(asyncio.CancelledError):
        with tl.pending_fill_poll(2, pending):
            with tl.pending_fill_poll(2, pending):
                assert tl.fill_poll_active(2, "153130")
                assert not tl.fill_poll_active(1, "153130")
            assert tl.fill_poll_active(2, "153130")
            raise asyncio.CancelledError()
    assert not tl.fill_poll_active(2, "153130")
