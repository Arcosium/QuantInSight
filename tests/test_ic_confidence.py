"""IC → 확신도 변환 — 에이전트 과거 예측력(IC)을 사이징/Ω 확신도로 (2026-06-15 ROI#2).

이미 있는 scorecard.db/IC 인프라를 '결정'에 배선. 과거에 잘 맞춘 신호(높은 IC)는 확신도↑,
역사적으로 틀린 신호(음수 IC)는 10점을 줘도 확신도↓. 표본 부족이면 중립(과신 방지).
"""
from tools.agent_scorecard import confidence_from_ic


def test_insufficient_sample_is_neutral():
    assert confidence_from_ic(0.3, n=5, min_n=20) == 0.5
    assert confidence_from_ic(None, n=200) == 0.5


def test_strong_positive_ic_high_confidence():
    c = confidence_from_ic(0.15, n=300)
    assert c > 0.7


def test_negative_ic_low_confidence():
    c = confidence_from_ic(-0.15, n=300)
    assert c < 0.4


def test_small_sample_shrinks_toward_neutral():
    # 표본이 막 임계 넘은 경우 중립 쪽으로 축소 → 큰 표본보다 덜 극단적
    near = confidence_from_ic(0.20, n=25, min_n=20, full_n=100)
    far = confidence_from_ic(0.20, n=200, min_n=20, full_n=100)
    assert abs(near - 0.5) < abs(far - 0.5)


def test_bounded_0_1():
    for ic in (-1.0, -0.3, 0.0, 0.3, 1.0):
        c = confidence_from_ic(ic, n=500)
        assert 0.0 <= c <= 1.0
