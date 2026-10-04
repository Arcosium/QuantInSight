"""Regression for the 2026-05-26 incident: logging in as hh0908 (mock, uid=2) hijacked
the global active account away from hh09080 (admin, uid=1) and cross-contaminated state.
Phase 2 makes each uid's broker/credentials/paths independent — a second login must not
touch the first user's context."""
from infra.user_context import UserRegistry
from infra import user_paths

_CREDS = {
    1: {"id": 1, "username": "hh09080", "is_admin": True, "kis_app_key": "K1",
        "kis_app_secret": "S1", "kis_account_no": "111-01",
        "kis_base_url": "https://openapi.koreainvestment.com:9443", "dart_key": "", "label": "a"},
    2: {"id": 2, "username": "hh0908", "is_admin": False, "kis_app_key": "K2",
        "kis_app_secret": "S2", "kis_account_no": "222-01",
        "kis_base_url": "https://openapivts.koreainvestment.com:29443", "dart_key": "", "label": "b"},
}


def test_second_login_does_not_hijack_first(monkeypatch):
    import infra.user_context as ucm
    monkeypatch.setattr(ucm.auth_store, "get_user_credentials", lambda u: _CREDS.get(int(u)))
    reg = UserRegistry()
    c1 = reg.get_or_create(1)
    b1 = c1.broker
    # uid=2 "logs in" → builds its own context; must not mutate c1
    c2 = reg.get_or_create(2)
    assert c1.broker is b1                      # uid=1 broker unchanged
    assert c1.broker.app_key == "K1"            # still hh09080's key
    assert c2.broker.app_key == "K2"            # hh0908 isolated
    assert c1.broker.is_mock is False and c2.broker.is_mock is True
    assert str(user_paths.equity_path(1)).endswith("/1/equity_curve.json")
    assert str(user_paths.equity_path(2)).endswith("/2/equity_curve.json")
    assert c1.broker._token_path != c2.broker._token_path   # per-uid token cache
