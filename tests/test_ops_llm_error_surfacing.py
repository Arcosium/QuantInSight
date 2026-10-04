"""ops_support 워커 LLM 예외 표면화 회귀 테스트.

버그(2026-05-21 밤 로그, cycle 41): llm_propose 가 321KB 프롬프트로 180초
타임아웃 → fail-soft 로 '변경 없음' 처리됐으나, 예외가 logger.error 로만 남고
record_error 를 안 거쳐 claude_response.json 에 type="error" 로 영속화되지 않았다.
그 결과 다음 사이클 운용지원실장이 "지난 사이클 진단 실패"를 인지하지 못했다.
(error_log.py 의 존재 이유가 바로 TimeoutError 같은 빈-메시지 예외의 표면화다.)

요구 동작: LLM 호출 예외 시 record_error 로 표면화하고, 안전하게 폴백한다.
"""
import asyncio

import infra.ops_support_worker as w


def _patch_boom(monkeypatch):
    calls = []
    # raising=False — RED 단계(아직 record_error 미도입)에서도 attribute 생성 허용.
    monkeypatch.setattr(w, "record_error",
                        lambda *a, **k: calls.append((a, k)), raising=False)
    async def boom(**kwargs):
        raise TimeoutError()
    monkeypatch.setattr(w, "chat_completion", boom)
    return calls


def test_llm_propose_surfaces_exception_via_record_error(monkeypatch):
    calls = _patch_boom(monkeypatch)
    res = asyncio.run(w.llm_propose("프롬프트", role="ops_support"))
    assert res == {}, "예외 시 빈 dict 로 안전 폴백"
    assert calls, "LLM 호출 예외가 record_error 로 표면화돼야 한다"
