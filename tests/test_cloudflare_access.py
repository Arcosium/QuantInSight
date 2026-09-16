import time
from types import SimpleNamespace
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from infra import cloudflare_access as access


@pytest.fixture
def signed(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cfg = {"team_domain":"unit-test.cloudflareaccess.com", "audience":"test-audience",
           "owner_email":"owner@example.test", "owner_uid":12345}
    monkeypatch.setattr(access,"settings",lambda:cfg)
    monkeypatch.setattr(access,"key_client",lambda _:SimpleNamespace(
        get_signing_key_from_jwt=lambda _:SimpleNamespace(key=key.public_key())))
    def make(**override):
        now = int(time.time())
        payload = {"iss":"https://unit-test.cloudflareaccess.com", "aud":"test-audience",
                   "sub":"test-owner", "email":"owner@example.test", "iat":now,"exp":now+120}
        return jwt.encode(payload | override,key,algorithm="RS256")
    return make


def test_valid_owner(signed):
    assert access.verify(signed())[0] == 12345


@pytest.mark.parametrize("change", [{"aud":"wrong"},{"iss":"https://wrong.cloudflareaccess.com"},
                                   {"email":"someone@example.test"},{"exp":1}])
def test_claims_are_verified(signed,change):
    with pytest.raises(PermissionError):
        access.verify(signed(**change))


def test_signature_is_verified(signed):
    token=signed();parts=token.split('.');parts[2]='AAAA'
    with pytest.raises(PermissionError):access.verify('.'.join(parts))


def test_origin_does_not_trust_old_session_or_email_header():
    from server.app import app
    client=TestClient(app)
    for path in ('/','/api/me','/api/login','/api/register','/legacy'):
        r=client.get(path,headers={"X-Session":"old-session", "Cf-Access-Authenticated-User-Email":"owner@example.test"})
        assert r.status_code == 401


def test_signed_http_and_csrf(signed):
    from server.app import app
    client=TestClient(app)
    headers={"Cf-Access-Jwt-Assertion":signed()}
    assert client.get('/api/auth_status',headers=headers).status_code == 200
    assert client.post('/api/start',headers=headers | {"Origin":"https://attacker.invalid"},json={}).status_code == 403


def test_no_local_auth_routes():
    from server.app import app
    routes={r.path for r in app.routes}
    assert not routes.intersection({'/api/login','/api/register','/api/recover_id',
                                   '/api/recover_password','/api/profile/password','/api/admin/members'})


def test_websocket_rejects_legacy_query_token():
    from server.app import app
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as err:
        with TestClient(app).websocket_connect('/ws?token=old-session'):
            pass
    assert err.value.code == 4401


def test_websocket_rejects_foreign_origin_even_with_valid_jwt(signed):
    from server.app import app
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as err:
        with TestClient(app).websocket_connect('/ws',headers={
            "Origin":"https://attacker.invalid","Cf-Access-Jwt-Assertion":signed()}):
            pass
    assert err.value.code == 4403
