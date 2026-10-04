"""_poll_fills_until_confirmed — '확정 후 증가(반복 폴링)' 동작 테스트.

사장 지시 2026-05-21: 주문 접수 직후 '즉시 체결'이 확인된 주문만 실행부에서 누적
카운트 +1 한다. 접수만 되고 체결이 미확인인 주문은 그 시점부터 5분마다 반복 폴링하며,

  - 체결이 확인되면 → 그때 누적 카운트 +1 + trade_log 기록 + 통신로그 '체결 확인됨'
    (type=trade_executed → 모바일 '체결 완료' 알림).
  - 아직 미체결이면 → 채팅 메시지도, 카운트도 올리지 않고 조용히 다음 주기 재시도.
  - 해당 시장이 마감되면(KIS가 미체결 지정가를 자동 취소) → 그 주문 폴링을 조용히 종료.

이전 구현(_reverify_fills)은 US 주문을 접수만으로 낙관적 +1 한 뒤 1회만 차감(rollback)
했다. 새 구현은 애초에 확정 전엔 안 올리므로 차감 자체가 없다(더 단순·정확).
"""
import asyncio

import pytest

import main_swarm
from main_swarm import ArquantOrchestrator


class _FakeBroker:
    def __init__(self, kr_after, us_after):
        self._kr = list(kr_after)
        self._us = list(us_after)
        self._acct_snap = None

    async def kr_holdings(self):
        return list(self._kr)

    async def _overseas_holdings(self):
        return list(self._us)


def _orch(broker, *, trades_executed=0, trade_log=None):
    o = object.__new__(ArquantOrchestrator)  # __init__ 우회 — 무거운 초기화 회피
    o.broker = broker
    o.uid = 1  # Phase 2: self._emit 가 _broadcast(msg, uid=self.uid) 로 라우팅하므로 필요
    o._trades_executed = trades_executed
    o._trade_log = list(trade_log or [])
    o._stop_event = asyncio.Event()
    return o


@pytest.fixture(autouse=True)
def _fast_and_silent(monkeypatch):
    # 5분 대기 seam 제거 + broadcast 캡처 + 폴링 상한 축소(무한루프 방지)
    monkeypatch.setattr(main_swarm, "_REVERIFY_DELAY_SEC", 0)
    monkeypatch.setattr(main_swarm, "_POLL_MAX_ATTEMPTS", 3)
    captured = []

    async def _fake_broadcast(ev, uid=None):
        captured.append(ev)

    monkeypatch.setattr(main_swarm, "_broadcast", _fake_broadcast)
    # 기본은 '장중'으로 — 마감 종료 조건을 명시적으로만 발동시키기 위함
    monkeypatch.setattr(main_swarm, "get_current_session", lambda: "US_TRADING")
    return captured


def test_unfilled_stays_silent_and_uncounted(_fast_and_silent):
    """장중 미체결: 카운트도 메시지도 올라가지 않아야 한다 (상한까지 폴링 후 종료)."""
    broker = _FakeBroker(kr_after=[], us_after=[])  # OXY 끝까지 미보유
    o = _orch(broker, trades_executed=0)
    pending = [{"ticker": "OXY", "side": "buy", "qty": 8}]

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=[]))

    assert o._trades_executed == 0, "미체결은 카운트되면 안 된다"
    assert o._trade_log == []
    assert _fast_and_silent == [], "미체결 시 채팅창에 아무 메시지도 띄우면 안 된다"


def test_fill_confirmed_increments_and_announces(_fast_and_silent):
    """폴링 중 체결 확인: 그때 +1 + trade_log + '체결 확인됨' trade_executed 1건."""
    broker = _FakeBroker(kr_after=[], us_after=[{"code": "XOM", "qty": 3}])  # 폴링 시 보유 확인됨
    o = _orch(broker, trades_executed=0)
    pending = [{"ticker": "XOM", "side": "buy", "qty": 3}]

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=[]))

    assert o._trades_executed == 1, "체결 확인 시 누적 카운트가 1 올라야 한다"
    assert len(o._trade_log) == 1 and o._trade_log[0]["ticker"] == "XOM"
    fills = [e for e in _fast_and_silent if e.get("type") == "trade_executed"]
    assert len(fills) == 1, "체결 확인 시 trade_executed(체결완료 알림) 1건만 나가야 한다"
    assert "체결 확인" in fills[0]["message"]
    assert fills[0]["trades_total"] == 1


def test_market_closed_emits_close_message(monkeypatch):
    """해당 시장 마감 시: 미체결분 폴링 종료 + 종결 메시지 1회 발화(상태 닫기). 체결 카운트는 0.
    사장 지시 2026-06-16: 기존 '조용히 종료'는 '접수 후 확인 중'이 영영 안 닫혀(uid2 375500)
    사용자가 상태를 알 수 없었다 → 종결 메시지를 명확히 띄운다(체결로 카운트하지는 않는다)."""
    monkeypatch.setattr(main_swarm, "_REVERIFY_DELAY_SEC", 0)
    monkeypatch.setattr(main_swarm, "_POLL_MAX_ATTEMPTS", 99)  # 상한이 아니라 '마감'으로 종료됨을 확인
    captured = []

    async def _fake_broadcast(ev, uid=None):
        captured.append(ev)

    monkeypatch.setattr(main_swarm, "_broadcast", _fake_broadcast)
    monkeypatch.setattr(main_swarm, "get_current_session", lambda: "OFF_HOURS")  # US 마감

    broker = _FakeBroker(kr_after=[], us_after=[])
    o = _orch(broker, trades_executed=0)

    asyncio.run(o._poll_fills_until_confirmed(
        [{"ticker": "OXY", "side": "buy", "qty": 8}], baseline_holdings=[]))

    assert o._trades_executed == 0, "미체결은 체결로 카운트하지 않는다"
    closes = [e for e in captured if e.get("type") == "trade_failed" and "OXY" in str(e.get("message", ""))]
    assert len(closes) == 1, "마감 종료 시 종결 메시지를 정확히 1회 띄워 상태를 닫아야 한다"
    assert ("종료" in closes[0]["message"]) or ("취소" in closes[0]["message"])


def test_poll_close_message_formats():
    m = main_swarm._poll_close_message("375500", "sell", 63, "market_close")
    assert "375500" in m and "63" in m and ("마감" in m or "취소" in m)
    m2 = main_swarm._poll_close_message("AAPL", "buy", 2, "max_attempts")
    assert "AAPL" in m2 and "종료" in m2


def test_baseline_prevents_false_positive(_fast_and_silent):
    """이미 5주 보유 중 3주 추가 매수 미체결: baseline(5)==after(5) → 체결로 오판하면 안 된다."""
    broker = _FakeBroker(kr_after=[], us_after=[{"code": "XOM", "qty": 5}])
    o = _orch(broker, trades_executed=0)
    pending = [{"ticker": "XOM", "side": "buy", "qty": 3}]
    baseline = [{"code": "XOM", "qty": 5}]  # 주문 직전 이미 5주

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=baseline))

    assert o._trades_executed == 0, "보유 변동이 없으면(미체결) 카운트되면 안 된다"
    assert [e for e in _fast_and_silent if e.get("type") == "trade_executed"] == []


# ── KR/US 비대칭 버그 수정 (2026-05-29): US 비동기 체결도 thesis 기록 ──────────────
# 기존엔 thesis 가 동기 실행부 `if filled:` 에서만 기록돼 US(폴링 확정)는 영원히 누락됐다.
# → 펀드기획팀장이 매도 직전 상기시킬 thesis 가 0건 → 사후관리실장 무계획 단타 차단 실패.
# 수정: 폴링 확정 경로에서도 매수 체결 시 _record_buy_thesis 를 호출한다.

def test_poll_buy_fill_records_thesis(_fast_and_silent):
    """US 비동기 매수가 폴링에서 체결 확인되면 펀드기획팀장 thesis 가 기록돼야 한다."""
    broker = _FakeBroker(kr_after=[],
                         us_after=[{"code": "XOM", "qty": 3, "avg_price": 100.0, "cur_price": 101.0}])
    o = _orch(broker, trades_executed=0)
    recorded = {}

    async def _fake_record(rec, cyc):
        recorded["rec"] = rec

    o._record_buy_thesis = _fake_record
    cyc = type("C", (), {"quant_report": "q", "news_report": "n", "order_obj": {}})()
    pending = [{"ticker": "XOM", "side": "buy", "qty": 3}]

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=[], cyc=cyc))

    assert recorded.get("rec"), "폴링 체결 확인 시 _record_buy_thesis 가 호출돼야 한다"
    assert recorded["rec"]["ticker"] == "XOM"
    assert recorded["rec"]["side"] == "buy" and recorded["rec"]["ok"] is True


def test_poll_sell_fill_does_not_record_thesis(_fast_and_silent):
    """매도 체결은 thesis 를 기록하지 않는다 (thesis 는 진입 시점에만)."""
    broker = _FakeBroker(kr_after=[], us_after=[])  # 3주 → 0주: 매도 체결
    o = _orch(broker, trades_executed=0)
    called = {"n": 0}

    async def _fake_record(rec, cyc):
        called["n"] += 1

    o._record_buy_thesis = _fake_record
    cyc = type("C", (), {"quant_report": "", "news_report": "", "order_obj": {}})()
    pending = [{"ticker": "XOM", "side": "sell", "qty": 3}]
    baseline = [{"code": "XOM", "qty": 3, "avg_price": 100.0, "cur_price": 101.0}]

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=baseline, cyc=cyc))

    assert called["n"] == 0, "매도 체결에는 thesis 기록을 호출하면 안 된다"


def test_poll_without_cyc_still_works(_fast_and_silent):
    """cyc 미전달(기존 호출부 호환): thesis 기록은 생략하되 체결 카운트는 정상."""
    broker = _FakeBroker(kr_after=[], us_after=[{"code": "XOM", "qty": 3}])
    o = _orch(broker, trades_executed=0)
    pending = [{"ticker": "XOM", "side": "buy", "qty": 3}]

    asyncio.run(o._poll_fills_until_confirmed(pending, baseline_holdings=[]))

    assert o._trades_executed == 1


def test_record_buy_thesis_skips_if_already_exists(monkeypatch):
    """중복가드: 이미 thesis 가 있으면 LLM 호출도 재기록도 하지 않는다
    (동기/폴링 양 경로가 같은 종목을 중복 기록하지 않게)."""
    import infra.position_thesis as pt
    o = object.__new__(ArquantOrchestrator)
    o.uid = 1
    called = {"think": 0}

    class _FP:
        async def think(self, p):
            called["think"] += 1
            return "목표가: 110\n손절가: 95\n계획 보유기간: 48h\n진입 사유 요약: x"

    o.fund_planner = _FP()
    monkeypatch.setattr(pt, "get", lambda uid, code: {"entry_ts": "기존"})  # 이미 존재
    recorded = []
    monkeypatch.setattr(pt, "record", lambda uid, code, th: recorded.append(code))
    rec = {"ticker": "XOM", "side": "buy", "ok": True,
           "fill_price": 100.0, "fill_currency": "USD", "avg_cost": 100.0}
    cyc = type("C", (), {"quant_report": "", "news_report": "", "order_obj": {}})()

    asyncio.run(o._record_buy_thesis(rec, cyc))

    assert called["think"] == 0, "이미 thesis 있으면 LLM 호출 skip"
    assert recorded == [], "이미 thesis 있으면 재기록 skip"
