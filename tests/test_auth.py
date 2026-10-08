import secrets
OWNER_PASSWORD = secrets.token_urlsafe(24)
MEMBER_PASSWORD = secrets.token_urlsafe(24)
WRONG_PASSWORD = secrets.token_urlsafe(24)
TEST_KEY = secrets.token_urlsafe(24)

import hashlib
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from autofolio import auth

HEADERS = {'X-Requested-With': 'QuantInSight'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('QUANTINSIGHT_AUTH_DIR', str(tmp_path / 'auth'))
    auth.bootstrap_admin('owner', OWNER_PASSWORD)
    app = FastAPI()
    app.include_router(auth.router)
    app.add_middleware(auth.AuthMiddleware)

    @app.get('/api/admin')
    def admin(request: Request):
        return auth.admin_required(request)

    @app.get('/')
    def home():
        return {'ok': True}

    with TestClient(app) as instance:
        yield instance


def log_in(client, username='owner', password=OWNER_PASSWORD):
    return client.post('/api/auth/login', headers=HEADERS, json={'username': username, 'password': password})


def register(client, name):
    return client.post('/api/auth/register', headers=HEADERS,
                       json={'username': name, 'password': MEMBER_PASSWORD})


def test_gate_cookie_and_revocation(client):
    assert client.get('/api/auth/me').status_code == 401
    assert client.get('/', follow_redirects=False).headers['location'] == '/login'
    response = log_in(client)
    assert response.status_code == 200
    assert 'HttpOnly' in response.headers['set-cookie']
    assert 'SameSite=lax' in response.headers['set-cookie']
    token = client.cookies[auth.COOKIE]
    with auth.connect() as db:
        stored = db.execute('SELECT token FROM sessions').fetchone()[0]
        assert stored == hashlib.sha256(token.encode()).hexdigest()
    assert client.get('/api/admin').status_code == 200
    assert client.post('/api/auth/logout', headers=HEADERS).status_code == 200
    client.cookies.set(auth.COOKIE, token)
    assert client.get('/api/auth/me').status_code == 401


def test_role_cannot_be_chosen_and_password_hashed(client):
    response = client.post('/api/auth/register', headers=HEADERS,
                           json={'username': 'intruder', 'password': MEMBER_PASSWORD, 'role': 'admin'})
    assert response.status_code == 422
    assert register(client, 'member').status_code == 200
    assert log_in(client, 'member', MEMBER_PASSWORD).json()['role'] == 'user'
    assert client.get('/api/admin').status_code == 403
    with auth.connect() as db:
        stored = db.execute('SELECT password FROM users WHERE username=?', ('member',)).fetchone()[0]
    assert MEMBER_PASSWORD not in stored
    assert auth.password_matches(MEMBER_PASSWORD, stored)
    assert log_in(client, 'member', WRONG_PASSWORD).status_code == 401


def test_csrf_and_rate_limit(client):
    payload = {'username': 'owner', 'password': OWNER_PASSWORD}
    assert client.post('/api/auth/login', json=payload).status_code == 403
    assert client.post('/api/auth/login', headers={**HEADERS, 'Origin': 'https://other.test'}, json=payload).status_code == 403
    with auth.connect() as db:
        bucket = hashlib.sha256(b'login:testclient').hexdigest()
        db.execute('INSERT INTO attempts VALUES(?,?,?)', (bucket, 30, time.time()+600))
    assert log_in(client).status_code == 429


def test_credentials_encrypted_and_scoped(client):
    register(client, 'member')
    log_in(client)
    value = {'provider': 'openai', 'model': 'test-model', 'api_key': TEST_KEY}
    response = client.post('/api/auth/credentials', headers=HEADERS, json=value)
    assert response.status_code == 200
    assert 'api_key' not in response.json()
    assert auth.get_credentials(1)['api_key'] == value['api_key']
    with auth.connect() as db:
        assert value['api_key'].encode() not in db.execute('SELECT secret FROM credentials').fetchone()[0]
    client.post('/api/auth/logout', headers=HEADERS)
    log_in(client, 'member', MEMBER_PASSWORD)
    assert client.get('/api/auth/credentials').json() == {'configured': False}
    client.delete('/api/auth/credentials', headers=HEADERS)
    assert auth.get_credentials(1)['api_key'] == value['api_key']
    assert client.post('/api/auth/credentials', headers=HEADERS, json={**value, 'provider': 'custom-url'}).status_code == 422


def test_expired_session_and_secure_remote_cookie(client):
    response = client.post('https://quantinsight.example/api/auth/login', headers=HEADERS,
                           json={'username': 'owner', 'password': OWNER_PASSWORD})
    assert '; Secure' in response.headers['set-cookie']
    log_in(client)
    with auth.connect() as db:
        db.execute('UPDATE sessions SET expires=?', (time.time()-1,))
    assert client.get('/api/auth/me').status_code == 401


def test_signup_optional_connections_are_encrypted_and_scoped(client):
    kis = {'app_key': 'example-app-key', 'app_secret': 'example-app-secret',
           'account_no': '12345678', 'account_product': '01', 'mode': 'live'}
    tf = {'username': 'example-timefolio', 'password': 'example-tf-password'}
    response = client.post('/api/auth/register', headers=HEADERS,
        json={'username': 'connected', 'password': MEMBER_PASSWORD, 'kis': kis, 'timefolio': tf})
    assert response.status_code == 200
    login_result = log_in(client, 'connected', MEMBER_PASSWORD).json()
    uid = login_result['id']
    assert auth.get_connection(uid, 'kis-live') == kis
    assert auth.get_connection(uid, 'timefolio') == tf
    assert auth.get_connection(1, 'kis-live') is None
    with auth.connect() as db:
        stored = b''.join(row['secret'] for row in db.execute('SELECT secret FROM connections'))
    assert b'example-app-secret' not in stored
    assert b'12345678' not in stored
    assert b'example-tf-password' not in stored
    metadata = client.get('/api/auth/connections').json()
    assert metadata == {'connections': [{'kind': 'kis-live', 'configured': True}, {'kind': 'timefolio', 'configured': True}]}
    assert client.post('/api/auth/connections', headers=HEADERS,
        json={'kis': {**kis, 'mode': 'live'}}).status_code == 200
    assert all(item['configured'] for item in client.get('/api/auth/connections').json()['connections'])
    assert auth.get_connection(uid, 'kis-live')['mode'] == 'live'
    client.post('/api/auth/logout', headers=HEADERS)
    log_in(client)
    assert not any(item['configured'] for item in client.get('/api/auth/connections').json()['connections'])


def test_partial_and_malformed_connections_rejected_without_creating_user(client):
    base = {'username': 'partial', 'password': MEMBER_PASSWORD}
    for extra in [{'kis': {'app_key': 'partial-key'}}, {'timefolio': {'username': 'partial'}},
                  {'kis': {'app_key': 'key', 'app_secret': 'secret', 'account_no': '1234567x',
                           'account_product': '01', 'mode': 'paper'}}, {'kis': {'mode': 'other'}}]:
        assert client.post('/api/auth/register', headers=HEADERS, json={**base, **extra}).status_code == 422
    with auth.connect() as db:
        assert db.execute('SELECT id FROM users WHERE username=?', ('partial',)).fetchone() is None
    response = client.post('/api/auth/register', headers=HEADERS,
        json={**base, 'kis': {'app_key': '', 'app_secret': '', 'account_no': '', 'account_product': ''},
              'timefolio': {'username': '', 'password': ''}})
    assert response.status_code == 200
    uid = log_in(client, 'partial', MEMBER_PASSWORD).json()['id']
    assert auth.get_connection(uid, 'kis-paper') is None
    assert not any(item['configured'] for item in client.get('/api/auth/connections').json()['connections'])
