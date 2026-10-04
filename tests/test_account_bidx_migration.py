"""kis_account_no_bidx 컬럼·upsert·부팅 백필 멱등 회귀."""
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


def test_schema_has_kis_account_no_bidx(store):
    with store._connect() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()}
    assert "kis_account_no_bidx" in cols


def test_upsert_sets_account_bidx(store):
    uid = store.upsert_user("u1", "Passw0rd!!xx", "AK", "AS",
                            "5012345601", "https://openapi.koreainvestment.com:9443")
    with store._connect() as c:
        row = c.execute("SELECT kis_account_no_bidx FROM users WHERE id=?",
                         (uid,)).fetchone()
    assert row["kis_account_no_bidx"] == store.bidx("5012345601")


def test_migration_backfills_blank_account_bidx_idempotent(store):
    uid = store.upsert_user("u2", "Passw0rd!!xx", "AK", "AS",
                            "777888999", "https://openapi.koreainvestment.com:9443")
    with store._connect() as c:
        c.execute("UPDATE users SET kis_account_no_bidx='' WHERE id=?", (uid,))
    s1 = store.migrate_passwords_and_bidx()
    with store._connect() as c:
        v = c.execute("SELECT kis_account_no_bidx FROM users WHERE id=?",
                       (uid,)).fetchone()["kis_account_no_bidx"]
    assert v == store.bidx("777888999")
    assert s1.get("acct_bidx", 0) >= 1
    s2 = store.migrate_passwords_and_bidx()
    assert s2.get("acct_bidx", 0) == 0


def test_migration_skips_corrupt_account_enc_row_preserved(store):
    """corrupt kis_account_no_enc → acct_bidx 카운트 안됨, 행(kis_account_no_bidx='') 보존."""
    uid = store.upsert_user("u3", "Passw0rd!!xx", "AK", "AS",
                            "1112223334", "https://openapi.koreainvestment.com:9443")
    # kis_account_no_enc 를 유효하지 않은 Fernet 토큰으로 교체, bidx 초기화
    with store._connect() as c:
        c.execute(
            "UPDATE users SET kis_account_no_enc='not-a-valid-fernet-token', "
            "kis_account_no_bidx='' WHERE id=?",
            (uid,),
        )
    stats = store.migrate_passwords_and_bidx()
    # (a) acct_bidx 카운트 0 — 복호 실패로 백필 안됨
    assert stats.get("acct_bidx", 0) == 0
    # (b) 행 보존 — kis_account_no_bidx 여전히 '' (부분 쓰기·행 삭제 없음)
    with store._connect() as c:
        val = c.execute(
            "SELECT kis_account_no_bidx FROM users WHERE id=?", (uid,)
        ).fetchone()["kis_account_no_bidx"]
    assert (val or "") == ""
