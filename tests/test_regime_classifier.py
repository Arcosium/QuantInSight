"""레짐 신호 — LLM 매크로 판단을 0~1 de-risk score 로 수치화 (2026-06-15 ROI#3, LLM 기반).

사장 결정: 국면 '판단'은 LLM(글로벌리서치팀장)이 유연하게(예외대응). 여기선 그 LLM 권고
(주식%/현금%)를 켈리 분수·비중 상한이 먹을 수 있는 숫자로 변환만 한다. 별도 결정론 분류기 X.
"""
from tools.regime import regime_score_from_macro


def test_defensive_macro_high_derisk():
    r = regime_score_from_macro("📈 자산 배분 권고: 주식 5% / 채권 0% / 원자재 0% / 현금 95%")
    assert r["regime"] == "risk_off"
    assert r["score"] >= 0.6          # 현금 95% → 강한 de-risk


def test_aggressive_macro_low_derisk():
    r = regime_score_from_macro("📈 자산 배분 권고: 주식 65% / 채권 0% / 원자재 0% / 현금 35%")
    assert r["regime"] == "risk_on"
    assert r["score"] <= 0.4


def test_balanced_macro_neutral():
    r = regime_score_from_macro("자산 배분 권고: 주식 50% / 현금 50%")
    assert r["regime"] == "neutral"


def test_unparseable_is_neutral():
    r = regime_score_from_macro("특별한 비중 언급 없음")
    assert r["regime"] == "neutral"
    assert r["score"] == 0.5


def test_score_bounded():
    for txt in ("주식 100% / 현금 0%", "주식 0% / 현금 100%", "엉터리"):
        assert 0.0 <= regime_score_from_macro(txt)["score"] <= 1.0
