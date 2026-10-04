import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    import infra.auth_store as a
    for n, v in [("_DATA_DIR", tmp_path), ("_DB_PATH", tmp_path / "auth.db"),
                 ("_FERNET_KEY_PATH", tmp_path / ".fernet.key"),
                 ("_AUDIT_PATH", tmp_path / "audit.log"),
                 ("_INITED", False), ("_FERNET", None),
                 ("_FERNET_RAW", None), ("_BIDX_KEY", None)]:
        monkeypatch.setattr(a, n, v, raising=False)
    a.init()
    return a


def test_delete_user_removes_row_and_sessions(store):
    uid = store.upsert_user("u1", "Passw0rd!!xx", "AK", "AS",
                            "5012345601", "https://openapi.koreainvestment.com:9443")
    tok = store.create_session(uid)
    assert store.lookup_session(tok) == uid
    assert store.delete_user(uid) is True
    assert store.find_user_by_username("u1") is None
    assert store.lookup_session(tok) is None
    assert store.delete_user(uid) is False


def test_list_members_fields(store):
    store.upsert_user("admin", "Passw0rd!!xx", "AK", "AS", "1",
                       "https://openapivts.koreainvestment.com:29443", is_admin=True)
    store.upsert_user("live1", "Passw0rd!!xx", "AK2", "AS2", "2",
                       "https://openapi.koreainvestment.com:9443")
    ms = {m["username"]: m for m in store.list_members()}
    assert ms["admin"]["is_admin"] is True and ms["admin"]["is_mock"] is True
    assert ms["live1"]["is_admin"] is False and ms["live1"]["is_mock"] is False
    assert "created_at" in ms["live1"] and "last_login_at" in ms["live1"]


def test_change_password_requires_current(store):
    uid = store.upsert_user("u3", "OldPassw0rd!!", "AK", "AS", "9",
                            "https://openapi.koreainvestment.com:9443")
    with pytest.raises(ValueError):
        store.change_password(uid, "WRONG", "NewPassw0rd!!")
    with pytest.raises(ValueError):
        store.change_password(uid, "OldPassw0rd!!", "short")
    assert store.change_password(uid, "OldPassw0rd!!", "NewPassw0rd!!") is True
    assert store.verify_password("u3", "NewPassw0rd!!")


def test_update_credentials_recomputes_bidx(store):
    uid = store.upsert_user("u4", "Passw0rd!!xx", "AK", "AS", "111",
                            "https://openapi.koreainvestment.com:9443")
    store.update_credentials(uid, kis_account_no="222", kis_app_secret="AS2")
    with store._connect() as c:
        r = c.execute("SELECT kis_account_no_bidx, kis_app_secret_bidx "
                       "FROM users WHERE id=?", (uid,)).fetchone()
    assert r["kis_account_no_bidx"] == store.bidx("222")
    assert r["kis_app_secret_bidx"] == store.bidx("AS2")
    assert store.get_user_credentials(uid)["kis_account_no"] == "222"
