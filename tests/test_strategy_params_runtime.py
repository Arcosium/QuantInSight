"""사장 지시 2026-06-04: 전략 파라미터 확장 — main_swarm 측 배선 헬퍼(순수 함수).
spec: docs/superpowers/specs/2026-06-04-strategy-param-expansion-design.md

검증:
  - MIN_QUANT_SCORE 결정론 게이트: 점수 미달 target 제거(미달만), 점수 없는 target 은 제외.
  - 계량분석팀장 호출 프롬프트용 전략 파라미터 블록: 정규화 가중치 + 활성 필터만 표기.
"""
from main_swarm import filter_targets_by_score, format_strategy_param_block


def test_filter_drops_below_min_score():
    kept, dropped = filter_targets_by_score(["A", "B", "C"], {"A": 7, "B": 5, "C": 4}, 6)
    assert kept == ["A"]
    assert set(dropped) == {"B", "C"}


def test_filter_min_zero_keeps_all():
    kept, dropped = filter_targets_by_score(["A", "B"], {"A": 1, "B": 0}, 0)
    assert kept == ["A", "B"] and dropped == []


def test_filter_missing_score_is_rejected():
    kept, dropped = filter_targets_by_score(["A", "D"], {"A": 7}, 6)
    assert kept == ["A"] and dropped == ["D"]


def test_filter_boundary_equal_is_kept():
    kept, dropped = filter_targets_by_score(["A"], {"A": 6}, 6)
    assert kept == ["A"] and dropped == []


def test_param_block_no_qw_weights_after_deterministic_engine():
    # 2026-06-04 결정론 점수 엔진: QW 채점 가중치는 블록에서 제거(파이썬 QIW_*가 처리). 필터만 advisory.
    params = {
        "MAX_BUY_VOLATILITY_PCT": 0, "RSI_OVERBOUGHT_SKIP": 0, "MIN_ADX_FOR_BUY": 0,
        "REQUIRE_FOREIGN_NET_BUY": False, "MAX_PRICE_EXTENSION_PCT": 0, "MIN_QUANT_SCORE": 6,
    }
    block = format_strategy_param_block(params)
    assert "채점 가중치" not in block       # 가중치 줄 제거됨


def test_param_block_lists_active_filters_only():
    params = {
        "MAX_BUY_VOLATILITY_PCT": 40, "RSI_OVERBOUGHT_SKIP": 70, "MIN_ADX_FOR_BUY": 0,
        "REQUIRE_FOREIGN_NET_BUY": True, "MAX_PRICE_EXTENSION_PCT": 0, "MIN_QUANT_SCORE": 7,
    }
    block = format_strategy_param_block(params)
    assert "40" in block            # 변동성 상한 활성
    assert "외국인" in block          # 수급 요구 활성
    assert "RSI" in block            # 과매수 회피 활성
    # 비활성(0) 필터는 표기하지 않는다
    assert "이격" not in block        # MAX_PRICE_EXTENSION_PCT=0 → 미표기
    assert "ADX" not in block        # MIN_ADX_FOR_BUY=0 → 미표기
