import pytest
from infra import user_context as uc


def test_registry_isolates_two_uids(monkeypatch):
    creds_by_uid = {
        1: {"id": 1, "username": "hh09080", "is_admin": True,
            "kis_app_key": "K1", "kis_app_secret": "S1", "kis_account_no": "111-01",
            "kis_base_url": "https://openapi.koreainvestment.com:9443",
            "dart_key": "", "label": "admin"},
        2: {"id": 2, "username": "hh0908", "is_admin": False,
            "kis_app_key": "K2", "kis_app_secret": "S2", "kis_account_no": "222-01",
            "kis_base_url": "https://openapivts.koreainvestment.com:29443",
            "dart_key": "", "label": "mock"},
    }
    monkeypatch.setattr(uc.auth_store, "get_user_credentials",
                        lambda uid: creds_by_uid.get(int(uid)))

    reg = uc.UserRegistry()
    c1 = reg.get_or_create(1)
    c2 = reg.get_or_create(2)

    assert c1 is not c2
    assert c1.creds["kis_app_key"] == "K1"
    assert c2.creds["kis_app_key"] == "K2"
    assert c1.is_admin is True and c2.is_admin is False
    # Same uid returns the same cached context (no rebuild)
    assert reg.get_or_create(1) is c1


def test_unknown_uid_raises(monkeypatch):
    monkeypatch.setattr(uc.auth_store, "get_user_credentials", lambda uid: None)
    reg = uc.UserRegistry()
    with pytest.raises(ValueError):
        reg.get_or_create(999)
