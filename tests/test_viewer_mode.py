from types import SimpleNamespace

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    import infra.auth_store as auth_store

    for name, value in [
        ("_DATA_DIR", tmp_path),
        ("_DB_PATH", tmp_path / "auth.db"),
        ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
        ("_AUDIT_PATH", tmp_path / "audit.log"),
        ("_INITED", False),
        ("_FERNET", None),
        ("_FERNET_RAW", None),
        ("_BIDX_KEY", None),
    ]:
        monkeypatch.setattr(auth_store, name, value, raising=False)
    auth_store.init()
    return auth_store


def _create(store, username, mode="trading"):
    return store.upsert_user(
        username, "Passw0rd!!xx", "", "", "",
        "https://openapi.koreainvestment.com:9443", account_mode=mode,
    )


def test_viewer_targets_the_single_admin_and_can_upgrade(store):
    admin_uid = _create(store, "hh09080")
    viewer_uid = _create(store, "watcher", store.VIEWER_MODE)

    assert store.is_admin(admin_uid) is True
    assert store.is_viewer(viewer_uid) is True
    assert store.admin_view_uid() == admin_uid

    store.set_account_mode(viewer_uid, store.TRADING_MODE)
    assert store.is_viewer(viewer_uid) is False


@pytest.mark.asyncio
async def test_ws_routes_admin_events_to_viewer_connection(monkeypatch):
    from server import app as app_module

    class Socket:
        def __init__(self):
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, message):
            self.messages.append(message)

    manager = app_module.WS()
    socket = Socket()
    await manager.connect(socket, uid=20, view_uid=10, client="web")
    await manager.send_to_uid(10, {"type": "agent", "message": "report"})

    assert socket.messages == [{"type": "agent", "message": "report"}]


def test_read_uid_and_write_guard_for_viewer(monkeypatch):
    from fastapi import HTTPException
    from server import app as app_module

    request = SimpleNamespace(state=SimpleNamespace(user_id=20))
    monkeypatch.setattr(app_module.auth_store, "is_viewer", lambda uid: uid == 20)
    monkeypatch.setattr(app_module.auth_store, "admin_view_uid", lambda: 10)

    assert app_module._read_uid(request) == 10
    with pytest.raises(HTTPException) as exc:
        app_module._require_trading(request)
    assert exc.value.status_code == 403


def test_admin_can_control_trading_even_if_mode_is_stale_viewer(monkeypatch):
    from server import app as app_module

    request = SimpleNamespace(state=SimpleNamespace(user_id=1))
    monkeypatch.setattr(app_module.auth_store, "is_admin", lambda uid: uid == 1)
    monkeypatch.setattr(app_module.auth_store, "is_viewer", lambda uid: True)
    # 통합 계정(2026-07-20): _require_trading 은 활성 프로필 uid 를 반환한다. 라이브 DB
    # 토글 상태에 의존하지 않도록 프로필 해석을 고정(uid1=실전 프로필 활성)한다.
    monkeypatch.setattr(app_module.auth_store, "resolve_profile_uid", lambda uid: 1)
    monkeypatch.setattr(app_module.auth_store, "get_user_credentials", lambda uid: {
        "account_mode": "trading", "kis_app_key": "test", "kis_account_no": "test",
        "kis_base_url": "https://openapi.koreainvestment.com:9443"})

    assert app_module._require_trading(request) == 1


@pytest.mark.asyncio
async def test_admin_start_and_stop_endpoints_use_admin_uid(monkeypatch):
    from server import app as app_module

    request = SimpleNamespace(state=SimpleNamespace(user_id=1))
    calls = []

    monkeypatch.setattr(app_module.auth_store, "is_admin", lambda uid: uid == 1)
    monkeypatch.setattr(app_module.auth_store, "is_viewer", lambda uid: True)
    # 통합 계정(2026-07-20): start/stop 은 활성 프로필로 라우팅 — 라이브 토글 비의존 고정.
    monkeypatch.setattr(app_module.auth_store, "resolve_profile_uid", lambda uid: 1)
    monkeypatch.setattr(app_module.auth_store, "get_user_credentials", lambda uid: {
        "account_mode": "trading", "kis_app_key": "test", "kis_account_no": "test",
        "kis_base_url": "https://openapi.koreainvestment.com:9443"})

    async def fake_start(uid, directive=None):
        calls.append(("start", uid, directive))

    async def fake_stop(uid):
        calls.append(("stop", uid))

    monkeypatch.setattr(app_module, "_start_uid", fake_start)
    monkeypatch.setattr(app_module, "_stop_uid", fake_stop)

    assert (await app_module.start(app_module.Req(directive="check"), request))["message"]
    assert (await app_module.stop(request))["message"]
    assert calls == [("start", 1, "check"), ("stop", 1)]
