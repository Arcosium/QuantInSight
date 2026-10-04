"""정량 운용안 모드(2026-09-25): 실장은 매수|회피만 고르고, 재량 매도 근거는 레드플래그 키워드."""
import asyncio

import agents.committee as cmt


def test_chief_limited_to_buy_or_avoid(monkeypatch):
    calls = []

    async def fake_turn(role, brief, dialogue, ask, want_stance, want_claims=False, thinking=None,
                        name_override=None, persona_override=None, stances=None):
        calls.append((role, stances))
        return {"text": "위험 없음", "stance": cmt.BUY, "confidence": .7} if want_stance else {"text": "논거"}

    monkeypatch.setattr(cmt, "_turn", fake_turn)
    rep = cmt.build_report("005930", "삼성전자")
    _o, _d, chief, _llm = asyncio.run(cmt.deliberate_target(
        "005930", "삼성전자", rep, quant_score=None, quant_excerpt="", news_excerpt="",
        macro_view="", systematic=True))
    assert ("chief", (cmt.BUY, cmt.AVOID)) in calls and chief["stance"] == cmt.BUY


def test_risk_keywords():
    assert cmt.risk_keywords("제3자배정 유상증자 결정") == ["유상증자"]
    assert cmt.risk_keywords("SMA5 이탈, ADX 약세") == []
