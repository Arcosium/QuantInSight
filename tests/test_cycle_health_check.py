"""사이클 자기검증 — cycle_health_warnings.

배경(2026-05-27 진단): 체결률이 20%(25주문 중 5건)인데 metrics 는 전부 ok:true 로 기록돼
실패가 묻혔다. 거부(KIS reject)·전건 미접수 같은 '진짜 이상'은 텍스트에만 남지 말고
코드로 점검해 경고로 떠야 한다. 접수-후-미체결(US 폴링 대기/KR 지정가 미도달)은 정상
대기 상태이므로 경고하지 않는다(잡음 방지) — 그건 별도 카운트 지표로만 추적한다.
"""
from main_swarm import cycle_health_warnings


def test_rejected_order_raises_warning():
    rows = [{"ticker": "NVDA", "side": "buy", "qty": 1,
             "accepted": False, "filled": False}]
    warns = cycle_health_warnings(rows)
    assert warns, "거부된 주문은 경고로 떠야 한다(묻히면 안 됨)"
    assert any("NVDA" in w or "거부" in w for w in warns)


def test_all_filled_no_warning():
    rows = [{"ticker": "005930", "side": "buy", "qty": 1, "accepted": True, "filled": True}]
    assert cycle_health_warnings(rows) == []


def test_accepted_pending_is_not_a_warning():
    """접수-후-미체결은 정상 대기(폴링/지정가) — 경고로 띄우면 잡음."""
    rows = [{"ticker": "XOM", "side": "buy", "qty": 3, "accepted": True, "filled": False}]
    assert cycle_health_warnings(rows) == []


def test_all_unaccepted_flags_systemic_failure():
    rows = [
        {"ticker": "A", "side": "buy", "qty": 1, "accepted": False, "filled": False},
        {"ticker": "B", "side": "buy", "qty": 1, "accepted": False, "filled": False},
    ]
    warns = cycle_health_warnings(rows)
    assert any("전건" in w or "미접수" in w for w in warns)


def test_watch_orders_ignored():
    rows = [{"ticker": "C", "side": "buy", "qty": 1, "accepted": False, "filled": False, "watch": True}]
    assert cycle_health_warnings(rows) == []


def test_empty_no_warning():
    assert cycle_health_warnings([]) == []


def test_ext_hours_defer_only_is_no_warning():
    """사장 보고 2026-06-26: 시간외(NXT) '사전 보류'만 있는 사이클은 정규장 재시도 예정인 정상
    보류다(실행부에서 종목별 trade_failed 로 이미 발화) — 자기검증 경보를 0건으로 둬야 한다.
    이전엔 '전건 미접수 → 실행 경로·자격증명·잔고 점검' 거짓경보가 밤사이 3회 발생했다."""
    rows = [{"ticker": "153130", "side": "sell", "qty": 10, "accepted": False, "filled": False,
             "fill_note": "시간외 주문 사전 보류"},
            {"ticker": "132030", "side": "sell", "qty": 5, "accepted": False, "filled": False,
             "fill_note": "시간외 주문 사전 보류"}]
    assert cycle_health_warnings(rows) == [], "정상 시간외 보류만 있으면 거짓경보를 띄우면 안 된다"


def test_real_reject_among_ext_holds_still_alarms():
    """시간외 보류와 *실거부*가 섞이면, 실거부는 여전히 거부 경고로 떠야 한다(묻히면 안 됨)."""
    rows = [{"ticker": "153130", "side": "sell", "qty": 10, "accepted": False, "filled": False,
             "fill_note": "시간외 주문 사전 보류"},
            {"ticker": "005930", "side": "buy", "qty": 1, "accepted": False, "filled": False,
             "fill_note": "주문가능금액 초과"}]
    warns = cycle_health_warnings(rows)
    assert any("005930" in w for w in warns)
    assert any("전건 미접수" in w or "자격증명" in w for w in warns), \
        "접수 0건 + 실거부 존재 → 전건 미접수 경보는 유지"


def test_warning_includes_actual_reason():
    """사장 피드백 2026-06-16: '사유 점검 필요'로 떠넘기지 말고 실제 사유를 경고에 직접 담는다."""
    rows = [{"ticker": "005930", "side": "buy", "qty": 1, "accepted": True, "filled": True},
            {"ticker": "375500", "side": "sell", "qty": 63, "accepted": False, "filled": False,
             "fill_note": "매도가능수량 0", "result": "375500 매도 보류 — 결제/제도 잠금 추정"}]
    reject_warn = next(w for w in cycle_health_warnings(rows) if "375500" in w)
    assert "매도가능수량 0" in reject_warn     # 실제 사유가 메시지에 직접 보인다
    assert "사유 점검 필요" not in reject_warn  # 더 이상 책임 떠넘기기 문구 없음
