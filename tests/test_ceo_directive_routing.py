"""per-user CEO directive routing (tag routing + handoff).

사장 지시 2026-05-26: 운용지원실장은 ADMIN·일반 유저 모두 파라미터 튜닝이 역할이므로,
비관리자에게 '코드 변경 불가' 거부 메시지나 ADMIN 아이디(hh09080)를 출력하지 않는다.

These tests stay offline and deterministic.
"""
import asyncio
import pytest
import main_swarm
from main_swarm import ArquantOrchestrator
from infra.user_context import UserContext


@pytest.fixture(autouse=True)
def _isolate_from_live(monkeypatch, isolate_orchestrator_data):
    """이 테스트는 실제 uid 에 바인딩된 오케스트레이터로 ceo_directive 를 호출하므로, 라이브 부작용
    (워커 subprocess spawn·LLM 호출·param_overrides 변경·대시보드 broadcast·응답로그 기록)을 반드시
    차단한다. 검증 대상은 ceo_directive 의 '동기 반환(라우팅·문구)'뿐이다.
    회귀 2026-06-09: 이 격리가 없어 pytest 실행이 운영 데이터(uid 3 TAKE_PROFIT_PCT 등)를 오염시켰다."""
    async def _noop_broadcast(*a, **k):
        return None
    monkeypatch.setattr(main_swarm, "_broadcast", _noop_broadcast)
    monkeypatch.setattr(ArquantOrchestrator, "_spawn_ops_support_worker",
                        lambda self, *a, **k: None)
    async def _noop_persist(self, *a, **k):
        return None
    monkeypatch.setattr(ArquantOrchestrator, "_auto_persist_directive", _noop_persist)


def _ctx(uid, admin):
    return UserContext({"id": uid, "username": f"u{uid}", "is_admin": admin,
        "kis_app_key": "K", "kis_app_secret": "S", "kis_account_no": "1-01",
        "kis_base_url": "https://openapivts.koreainvestment.com:29443",
        "dart_key": "", "label": "x"})


def test_non_admin_ops_support_not_refused_and_no_admin_id_leak():
    """사장 지시 2026-05-26: 비관리자가 운용지원실장에게 코드/배포 단어를 보내도
    '공유 소스 코드를 변경할 수 없습니다' 류 거부나 ADMIN 아이디(hh09080)를 출력하지 않는다."""
    from main_swarm import ArquantOrchestrator
    o = ArquantOrchestrator(_ctx(2, admin=False))
    reply = asyncio.run(o.ceo_directive("@운용지원실장 소스 코드 고쳐서 배포해줘"))
    assert "공유 소스 코드를 변경할 수 없습니다" not in reply
    assert "hh09080" not in reply


def test_non_admin_ops_support_param_tuning_not_gated():
    """Non-admin param/strategy tuning (no code-mod intent) must NOT hit the
    code-modification refusal — it proceeds through the normal ops_support path."""
    from main_swarm import ArquantOrchestrator
    o = ArquantOrchestrator(_ctx(3, admin=False))
    reply = asyncio.run(o.ceo_directive("@운용지원실장 익절 비율을 좀 올려줘"))
    # The deterministic code-mod refusal sentence must NOT appear.
    assert "공유 소스 코드를 변경할 수 없습니다" not in reply


def test_admin_ops_support_not_code_gated():
    """Admin must NOT be hit by the deterministic code-modification refusal even
    when the message expresses code/deploy intent — admin retains ops_support behavior."""
    from main_swarm import ArquantOrchestrator
    o = ArquantOrchestrator(_ctx(1, admin=True))
    reply = asyncio.run(o.ceo_directive("@운용지원실장 소스 코드 고쳐서 배포해줘"))
    assert "공유 소스 코드를 변경할 수 없습니다" not in reply


def test_unknown_tag_does_not_error_out():
    """Wrong/unknown tag must hand off (to orchestrator) rather than returning the
    old hard '에이전트 없음' error string. We assert it does not return that error."""
    from main_swarm import ArquantOrchestrator
    o = ArquantOrchestrator(_ctx(4, admin=False))
    # No real LLM key resolves at think() time for a fabricated agent; the orchestrator
    # think() may return an error-prefixed string, but it must NOT be the '없음' rejection.
    reply = asyncio.run(o.ceo_directive("@존재하지않는팀장 안녕"))
    assert "가능:" not in reply or "존재하지않는팀장" not in reply
