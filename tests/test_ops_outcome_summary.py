"""운용지원실장 멀티사이클 집계(B 환각 차단) + 수익성 스코어카드 주입(F3), 2026-06-18.

B: OPS#460 가 '장중 다수 리스크 미승인'을 날조했으나 실제 반려는 하루 1건뿐, 나머지는 '주문 미생성'.
   결정론 집계(체결/반려/주문미생성)를 프롬프트에 주입해 '반려 폭풍' 날조를 차단한다.
F3: 원장 권위 실현 기대값(승률·평균손익·기대값·비용드래그)을 주입해 고회전 수익성을 보고 튜닝하게 한다.
"""
import json
from infra.ops_support_worker import summarize_recent_outcomes, build_prompt


def test_distinguishes_rejected_from_no_orders():
    cycles = [
        # 반려: 주문 생성됐으나 미승인/거부
        {"orders_executed": json.dumps([{"side": "buy", "ticker": "X", "accepted": False, "filled": False}]),
         "orders_planned": json.dumps([{"ticker": "X"}])},
        # 주문 미생성: 후보 0·계획 0 (오케스트레이터가 '없음' 선택)
        {"orders_executed": "[]", "orders_planned": "[]"},
        # 체결
        {"orders_executed": json.dumps([{"side": "buy", "ticker": "Y", "accepted": True, "filled": True}]),
         "orders_planned": json.dumps([{"ticker": "Y"}])},
        {"orders_executed": "[]", "orders_planned": "[]"},
    ]
    s = summarize_recent_outcomes(cycles)
    assert s["rejected_orders"] == 1
    assert s["executed_orders"] == 1
    assert s["no_order_cycles"] == 2


def test_empty_cycles_safe():
    s = summarize_recent_outcomes([])
    assert s == {"cycles": 0, "executed_orders": 0, "rejected_orders": 0, "no_order_cycles": 0}


def test_build_prompt_includes_outcome_tally_and_scorecard():
    ctx = {
        "target_cycle": {"started_at": "t", "session": "KR_TRADING"},
        "recent_cycles": [],
        "recent_outcomes": {"cycles": 7, "executed_orders": 1, "rejected_orders": 1, "no_order_cycles": 6},
        "realized": {"sell_count": 8, "win_count": 2, "win_rate": 25.0,
                     "total_realized_krw": -64581.0, "avg_win_krw": 500.0, "avg_loss_krw": -10000.0,
                     "expectancy_krw": -8072.0, "cost_drag_krw": 20831.0,
                     "kr": {"sell_count": 5, "realized": -43750.0}, "us": {"sell_count": 3, "realized_usd": -13.77}},
    }
    p = build_prompt(ctx)
    # B: 정확한 집계가 프롬프트에 들어가 '반려 폭풍' 날조를 막는다
    assert "반려 1" in p and "주문 미생성 6" in p
    # F3: 수익성 스코어카드(기대값·승률)가 들어간다
    assert "기대값" in p and "25.0" in p
