"""사장 지시 2026-06-04: ① 퀀트 점수가 선정에 영향 — 미달 제거 + 점수 내림차순 + 최대 종목수 캡.
점수 없는 종목은 매수 보류. 루브릭 블록은 PASS1 선정에 채점 기준 주입."""
from main_swarm import filter_targets_by_score, format_scoring_rubric_block


def test_drops_below_min_and_sorts_desc():
    kept, dropped = filter_targets_by_score(["A", "B", "C"], {"A": 7, "B": 9, "C": 4}, 6, max_names=8)
    assert kept == ["B", "A"]          # 9 > 7, C(4) 제거
    assert dropped == ["C"]


def test_caps_to_max_names_keeping_top():
    kept, dropped = filter_targets_by_score(
        ["A", "B", "C", "D"], {"A": 6, "B": 9, "C": 7, "D": 8}, 6, max_names=2)
    assert kept == ["B", "D"]          # 상위 2개(9,8)
    assert set(dropped) == {"A", "C"}  # 캡 초과분도 dropped 에 보고


def test_missing_score_is_rejected():
    kept, dropped = filter_targets_by_score(["A", "X"], {"A": 7}, 6, max_names=8)
    assert kept == ["A"] and dropped == ["X"]


def test_max_names_zero_means_no_cap():
    kept, _ = filter_targets_by_score(["A", "B", "C"], {"A": 7, "B": 8, "C": 9}, 6, max_names=0)
    assert set(kept) == {"A", "B", "C"}


def test_backward_compatible_default_args():
    # max_names 미지정 시 기존 동작(캡 없음, 미달만 제거) — 정렬은 적용.
    kept, dropped = filter_targets_by_score(["A", "B"], {"A": 5, "B": 7}, 6)
    assert kept == ["B"] and dropped == ["A"]


def test_rubric_block_describes_weights_and_gate():
    block = format_scoring_rubric_block(
        {"rsi": 5, "macd": 10, "mom": 12}, {"QUANT": 60, "NEWS": 25, "MACRO": 15}, 6)
    assert "퀀트점수" in block
    assert "6" in block                # MIN_QUANT_SCORE
    assert "모멘텀" in block or "MOM" in block  # 최상위 가중 지표 언급
