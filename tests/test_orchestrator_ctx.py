import pytest

from infra.user_context import UserContext


@pytest.fixture(autouse=True)
def _iso(isolate_orchestrator_data):
    """실 auth DB/데이터를 건드리지 않도록 격리(오케스트레이터 전체 생성 테스트)."""


def _ctx(uid, mock=False):
    return UserContext({
        "id": uid, "username": f"u{uid}", "is_admin": uid == 1,
        "kis_app_key": f"K{uid}", "kis_app_secret": f"S{uid}",
        "kis_account_no": f"{uid}-01",
        "kis_base_url": ("https://openapivts.koreainvestment.com:29443" if mock
                         else "https://openapi.koreainvestment.com:9443"),
        "dart_key": "", "label": f"u{uid}",
    })


def test_orchestrator_owns_uid_and_per_uid_paths():
    from main_swarm import ArquantOrchestrator
    o1 = ArquantOrchestrator(_ctx(1))
    o2 = ArquantOrchestrator(_ctx(2, mock=True))
    assert o1.uid == 1 and o2.uid == 2
    assert o1.broker is not o2.broker
    assert o1.broker.app_key == "K1" and o2.broker.app_key == "K2"
    assert str(o1.equity_path).endswith("/1/equity_curve.json")
    assert str(o2.equity_path).endswith("/2/equity_curve.json")
    # 로컬 LLM 전환 후 per-user LLM 키는 없음 — orchestrator.api_key 는 항상 빈 값.
    assert o1.orchestrator.api_key == ""
    assert o2.orchestrator.api_key == ""
