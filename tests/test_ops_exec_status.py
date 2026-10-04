"""운용지원실장에게 전달되는 직전 사이클 실행결과는 '접수—체결폴링중'과 '실패/반려'를 구분해야 한다.

버그 2026-05-22: 주문은 비동기 체결(접수 후 5분 폴링)인데, 운용지원실장이 사이클 종료
스냅샷의 filled=false 를 '실패'로 단정해 MAX_TRADES_PER_CYCLE·ENABLE_CHEAP_FALLBACK 를
잘못 변경했다(실제로는 직후 정상 체결). 실행결과 요약이 상태를 명확히 구분하고,
프롬프트가 '미체결≠실패' 가이드를 포함해야 한다.
"""
from infra.ops_support_worker import _summarize_exec_results, build_prompt


def test_pending_not_marked_failed():
    s = _summarize_exec_results([
        {"ticker": "017670", "side": "buy", "qty": 6, "accepted": True, "filled": False, "ok": False}])
    assert "017670" in s
    assert "폴링" in s
    assert "실패아님" in s


def test_filled_marked():
    s = _summarize_exec_results([
        {"ticker": "MSFT", "side": "sell", "qty": 1, "accepted": True, "filled": True, "ok": True}])
    assert "체결" in s


def test_rejected_marked():
    s = _summarize_exec_results([
        {"ticker": "005930", "side": "buy", "qty": 6, "accepted": False, "filled": False, "ok": False}])
    assert "반려" in s or "미접수" in s


def test_rejected_shows_reason_from_fill_note():
    # 사장 지시 2026-06-16: 미접수·반려는 원인(fill_note)을 함께 표시해 OPS 가 '왜 안 됐는지'를
    # 정확히 보고 환각(예: '유동성 부족') 대신 올바르게 판단하게 한다 — 241710 주문가능금액 부족 사례.
    s = _summarize_exec_results([
        {"ticker": "241710", "side": "buy", "qty": 7, "accepted": False, "filled": False,
         "ok": False, "fill_note": "KIS 주문가능금액 부족"}])
    assert "241710" in s
    assert "주문가능금액" in s


def test_empty():
    assert _summarize_exec_results([]) == "없음"
    assert _summarize_exec_results(None) == "없음"


def test_build_prompt_has_async_guidance():
    ctx = {"target_cycle": {"orders_executed": [
        {"ticker": "017670", "side": "buy", "qty": 6, "accepted": True, "filled": False, "ok": False}]}}
    p = build_prompt(ctx)
    assert "비동기" in p
    assert ("실패가 아" in p) or ("실패로 단정" in p)
