import json
import secrets
import sqlite3
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autofolio import auth, crypto_view


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('QUANTINSIGHT_AUTH_DIR', str(tmp_path / 'auth'))
    app = FastAPI()
    app.add_middleware(auth.AuthMiddleware)
    app.include_router(auth.router)
    app.include_router(crypto_view.router)
    password = secrets.token_urlsafe(24)
    auth.bootstrap_admin('owner', password)
    c = TestClient(app)
    c.post('/api/auth/login', headers={'X-Requested-With': 'QuantInSight'},
           json={'username': 'owner', 'password': password})
    return c


@pytest.fixture
def marks(tmp_path, monkeypatch):
    path = tmp_path / 'paper.sqlite3'
    monkeypatch.setenv('HYFE_PAPER_DB', str(path))
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE marks(ts INTEGER PRIMARY KEY, payload TEXT)')
        for ts, event, equal in [(3000, 1.12, 1.03), (1000, 1.0, 1.0), (2000, 1.05, 0.98)]:
            db.execute('INSERT INTO marks VALUES (?,?)', (ts, json.dumps({'books': {
                'event': {'net': event, 'funded': 7}, 'equal': {'net': equal, 'funded': 8}}})))
    return path


def upstream(mock, payload):
    mock.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()


def test_actual_fee_only_series_keeps_equity_units_and_variant_isolation(client, marks):
    before = marks.read_bytes()
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        upstream(call, {'equity': 1.12, 'settled_return_pct': 12, 'series': []})
        event = client.get('/crypto/api/book/cnn?history_limit=10&history_offset=0').json()
        equal = client.get('/crypto/api/book/cnn_equal').json()
    assert event['series'] == [{'ts': 1000, 'equity': 1}, {'ts': 2000, 'equity': 1.05}, {'ts': 3000, 'equity': 1.12}]
    assert (event['series'][-1]['equity'] - 1) * 100 == pytest.approx(event['settled_return_pct'])
    assert equal['series'][-1]['equity'] == 1.03
    assert event['series_basis'] == crypto_view.SERIES_BASIS
    assert marks.read_bytes() == before


def test_missing_database_is_not_created_or_filled(tmp_path, monkeypatch, client):
    missing = tmp_path / 'missing.sqlite3'
    monkeypatch.setenv('HYFE_PAPER_DB', str(missing))
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        upstream(call, {'equity': 1.15, 'series': []})
        data = client.get('/crypto/api/book/cnn').json()
    assert data['series'] == [] and data['series_message']
    assert not missing.exists()


def test_legacy_books_preserve_existing_series(client):
    payload = {'equity': 0.97, 'series': [{'ts': 1, 'equity': 0.97}]}
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call, patch('autofolio.crypto_view.paper_series') as series:
        upstream(call, payload)
        assert client.get('/crypto/api/book/v2').json() == payload
        series.assert_not_called()


@pytest.mark.parametrize('path', ['book/private', 'summary?url=http://elsewhere', 'candles/BTC?tf=16',
    'log?limit=201', 'alarms?limit=0', 'candles/BTC?n=401', 'book/cnn?history_offset=-1',
    'book/cnn?history_limit=10&history_limit=20'])
def test_endpoint_and_query_allowlist(client, path):
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        assert client.get('/crypto/api/' + path).status_code in (404, 422)
        call.assert_not_called()


def test_unicode_candles_fixed_origin(client):
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        upstream(call, {'candles': []})
        assert client.get('/crypto/api/candles/龙虾?tf=240&n=80').status_code == 200
        assert call.call_args.args[0] == 'http://127.0.0.1:8996/api/candles/%E9%BE%99%E8%99%BE?tf=240&n=80'


def test_auth_required_and_legacy_view_redirects(client):
    assert client.get('/crypto/view', follow_redirects=False).headers['location'] == '/?market=crypto&view=paper'
    assert client.get('/crypto/static/index.html').status_code == 404
    client.cookies.clear()
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        assert client.get('/crypto/api/summary', follow_redirects=False).status_code in (401, 303)
        call.assert_not_called()


def test_missing_upstream_book_is_unavailable(client):
    with patch('autofolio.crypto_view.urllib.request.urlopen') as call:
        upstream(call, None)
        assert client.get('/crypto/api/book/cnn').status_code == 503
