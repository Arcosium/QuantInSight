"""thesis 정리 로그 메시지 정확화 (2026-06-15).

버그: 보유 0이 된 코드의 thesis 를 정리할 때 무조건 "전량 매도로 thesis 제거"라 찍어,
실제로는 매도된 적 없는(보유 0 placeholder) DBC 같은 코드도 '매도'로 오인하게 만든다.
수정: 보유 0 = 매도일 수도/미보유일 수도 → 단정하지 않는 중립 문구.
"""
from main_swarm import _thesis_prune_msg


def test_message_does_not_claim_sale():
    msg = _thesis_prune_msg("commodity", ["DBC"])
    assert "전량 매도" not in msg
    assert "DBC" in msg


def test_message_says_pruned():
    msg = _thesis_prune_msg("펀드기획", ["012510"])
    assert "정리" in msg
    assert "펀드기획" in msg
