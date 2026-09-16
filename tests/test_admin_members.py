"""Cloudflare owner access replaces application membership management."""
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize("path",["/api/admin/members","/api/admin/member", "/api/admin/members/delete", "/api/admin/members/grant"])
def test_membership_routes_are_gone_even_for_authenticated_owner(path,monkeypatch):
    from server import app as mod
    monkeypatch.setattr(mod.cloudflare_access,"verify",lambda token:(12345,9999999999))
    c=TestClient(mod.app)
    assert c.get(path).status_code==404
    assert c.post(path,json={}).status_code==404
