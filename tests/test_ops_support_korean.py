"""운용지원실장(ops_support) 한국어 응답 강제 (사장 지시 2026-06-12).

로컬 LLM 가 영문 파라미터명 맥락에 끌려 근거(rationale)를 영어로 작성하던 사고
(uid1 cyc312 03:08 KST, 근거 전체 영어) 재발 방지 — 다른 에이전트(계량분석/프롭트레이딩)와
달리 ops_support 프롬프트엔 한국어 강제 지시가 빠져 있었다.
"""
from agents.specialists import create_ops_support


def test_ops_support_prompt_enforces_korean():
    agent = create_ops_support()
    assert "한국어" in agent.system_prompt


def test_ops_support_korean_applies_to_summary_and_rationale():
    # summary·rationale 두 자유서술 필드 모두 한국어여야 함을 명시한다.
    sp = create_ops_support().system_prompt
    assert "summary" in sp and "rationale" in sp
    # 한국어 강제 문구가 자유서술 필드를 지칭하도록(영문 식별자/숫자는 예외 허용).
    assert "한국어" in sp
