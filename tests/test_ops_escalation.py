"""ops 자동 에스컬레이션 — 사장 지시 2026-06-17.

배경(uid1 hh09080 06/16 야간): US 매수가 6사이클 연속 '주문가능금액 초과'로 거부됐는데,
운용지원실장(파라미터 튜닝 전용)은 원인을 '환율·수수료 갭/예산 과도'로 오진하고 예산 노브만
왔다갔다(thrashing). proposed_source_changes 는 매번 비어 코드 점검 필요를 한 번도 사장에게
에스컬레이션하지 못했다(LLM 이 changes 를 안 냄).

교정: 같은 사유의 주문 실패가 여러 사이클 반복되면 — 파라미터로 안 풀린다는 증거 — 결정론적으로
감지해 proposed_source_changes 로 승격하고 사장에게 코드 점검을 알린다(LLM 비의존).
"""
import json
from infra.ops_support_worker import detect_recurring_order_failure


def _cyc(cid, executed):
    return {"id": cid, "orders_executed": json.dumps(executed)}


def _us_fail(tk):
    return {"ticker": tk, "market": "US", "side": "buy", "accepted": False,
            "result": f"[US매수 실패] {tk} → 주문가능금액을 초과 했습니다"}


def test_escalates_when_same_failure_repeats_across_cycles():
    cycles = [_cyc(408, [_us_fail("PG"), _us_fail("ARKX")]),
              _cyc(410, [_us_fail("JNJ")]),
              _cyc(412, [_us_fail("XOM")])]
    msg = detect_recurring_order_failure(cycles, min_cycles=3)
    assert msg is not None
    assert "반복" in msg and "점검" in msg


def test_no_escalation_below_threshold():
    cycles = [_cyc(408, [_us_fail("PG")]), _cyc(410, [_us_fail("JNJ")])]
    assert detect_recurring_order_failure(cycles, min_cycles=3) is None


def test_no_escalation_when_buys_succeed():
    ok = [{"ticker": "DAL", "market": "US", "side": "buy", "accepted": True,
           "result": "[US매수] DAL → 주문 전송 완료 되었습니다"}]
    cycles = [_cyc(400, ok), _cyc(404, ok), _cyc(406, ok)]
    assert detect_recurring_order_failure(cycles, min_cycles=3) is None


def test_parses_orders_executed_json_string():
    # cycle_store 는 orders_executed 를 JSON 문자열로 저장 — 파싱해서 감지해야 함
    cycles = [_cyc(1, [_us_fail("A")]), _cyc(2, [_us_fail("B")]), _cyc(3, [_us_fail("C")])]
    assert detect_recurring_order_failure(cycles, min_cycles=3) is not None


def test_same_cycle_multiple_fails_counts_once_per_cycle():
    # 한 사이클에 2건 실패해도 '사이클 1개'로 — min_cycles 는 사이클 수 기준
    cycles = [_cyc(408, [_us_fail("PG"), _us_fail("ARKX")])]
    assert detect_recurring_order_failure(cycles, min_cycles=3) is None
