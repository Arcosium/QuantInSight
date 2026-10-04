"""Local registration and account recovery were retired with Access."""
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize("path",["/api/login","/api/register","/api/recover_id","/api/recover_password","/api/check_username"])
def test_retired_auth_endpoints(path,monkeypatch):
    from server import app as mod
    c=TestClient(mod.app)
    assert c.post(path,json={}).status_code==401
    monkeypatch.setattr(mod.cloudflare_access,"verify",lambda token:(12345,9999999999))
    assert c.post(path,json={}).status_code==404
