"""복구 인자 = 한투 계좌번호 + 한투 App Secret (2개)."""
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
    a.upsert_user("trader", "Passw0rd!!xx", "APPKEY1", "SECRET1",
                  "5012345601", "https://openapi.koreainvestment.com:9443")
    return a


def test_find_username_by_account_and_secret(store):
    assert store.find_username_by_factors("5012345601", "SECRET1") == "trader"


def test_find_fails_on_wrong_secret(store):
    assert store.find_username_by_factors("5012345601", "WRONG") is None


def test_find_fails_on_empty(store):
    assert store.find_username_by_factors("", "SECRET1") is None


def test_reset_password_two_factor(store):
    ok = store.reset_password_by_factors("trader", "5012345601", "SECRET1",
                                         "NewPassw0rd!!")
    assert ok is True
    assert store.verify_password("trader", "NewPassw0rd!!")


def test_reset_policy_checked_before_factor_match_enum_oracle(store):
    with pytest.raises(ValueError):
        store.reset_password_by_factors("trader", "WRONGACCT", "WRONGSEC", "short")


def test_reset_fails_wrong_factors_valid_policy(store):
    assert store.reset_password_by_factors("trader", "BAD", "BAD",
                                           "ValidPassw0rd!!") is False
