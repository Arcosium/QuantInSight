"""매도 잠김 에스컬레이션 알림 — 모의/실계정 라벨 분기 (2026-06-22).

배경: 모의서버는 '오늘 매도 누적(thdt_sll_qty)'으로 잔여 보유분의 ord_psbl_qty 를 0 으로 깎는
일일 카운터 quirk 가 있다(uid2 316140: hldg 100·ord_psbl 0·thdt_sll 160·펜딩 0). 익일
리셋 시 자동 해소된다. 그런데 잠김 에스컬레이션이 실계정용 '🚨 손절 매도 잠김 — 수동 확인 필요'
WARN 을 모의에도 띄워 오해를 샀다. 모의는 INFO + '일일 매도한도 제약(익일 해소)'로 순화하고,
실계정은 기존 WARN(진짜 결제/제도 잠금 가능)을 그대로 보존한다.
"""
from main_swarm import _locked_sell_escalation_alert


def test_live_keeps_stern_warn():
    sev, title, msg = _locked_sell_escalation_alert("007340", 50, 3, is_mock=False)
    assert sev == "WARN"
    assert title == "손절 매도 잠김 에스컬레이션"
    assert "수동 확인" in msg          # 실계정은 수동 확인 유도 보존
    assert "3사이클" in msg


def test_mock_softened_to_info():
    sev, title, msg = _locked_sell_escalation_alert("316140", 100, 3, is_mock=True)
    assert sev == "INFO"               # 모의는 경보 아님
    assert "모의" in title
    assert "일일 매도" in msg           # 모의서버 일일 매도한도 제약
    assert "익일" in msg               # 익일 자동 해소 안내
    assert "수동 확인" not in msg       # 모의는 수동조치 불필요(오해 방지)
