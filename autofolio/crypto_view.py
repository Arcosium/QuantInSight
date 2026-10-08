"""Authenticated operational data for the native QuantInSight paper trading view."""
import json
import math
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from .auth import user
from .config import HOME

router = APIRouter()
BOOKS = {'cnn', 'cnn_equal', 'v1', 'v2', 'v2d'}
SERIES_BASIS = '4시간 마감 · 수수료 반영 · 펀딩비 제외'


def paper_series(key):
    """Read only recorded fee-only NAV, keeping the upstream book's 1.0 units."""
    variant = {'cnn': 'event', 'cnn_equal': 'equal'}[key]
    database = Path(os.environ.get('HYFE_PAPER_DB', HOME / 'projects/HYFE_QTPA/work/paper_trade/paper.sqlite3'))
    try:
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            rows = db.execute('''SELECT ts, json_extract(payload, ?) FROM
                (SELECT ts, payload FROM marks ORDER BY ts DESC LIMIT 10000)
                WHERE json_valid(payload) ORDER BY ts''', (f'$.books.{variant}.net',)).fetchall()
    except (OSError, sqlite3.Error):
        return [], '저장된 마감 평가 기록을 불러오지 못했습니다.'
    series = [{'ts': ts, 'equity': value} for ts, value in rows
              if isinstance(ts, int) and isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    return series, '' if series else '아직 저장된 마감 평가 기록이 없습니다.'


def query_string(endpoint, query):
    if endpoint.startswith('book/'):
        rules = {'history_offset': (0, 10_000_000), 'history_limit': (1, 200)}
    elif endpoint.startswith('candles/'):
        rules = {'tf': (15, 1440), 'n': (30, 400)}
    elif endpoint in {'alarms', 'log'}:
        rules = {'limit': (1, 500 if endpoint == 'alarms' else 200)}
    else:
        rules = {}
    values = {}
    for name, raw in query.multi_items():
        if name not in rules or name in values:
            raise HTTPException(422, '지원하지 않거나 중복된 조회 조건입니다.')
        try:
            value = int(raw)
        except ValueError:
            raise HTTPException(422, '조회 조건은 정수여야 합니다.')
        low, high = rules[name]
        if not low <= value <= high or name == 'tf' and value not in (15, 60, 240, 1440):
            raise HTTPException(422, '조회 조건이 허용 범위를 벗어났습니다.')
        values[name] = value
    return urlencode(values)


@router.get('/crypto/view')
def view(request: Request):
    user(request)
    return RedirectResponse('/?market=crypto&view=paper', status_code=307)


@router.get('/crypto/api/{endpoint:path}')
def api(endpoint: str, request: Request):
    user(request)
    candle = endpoint.removeprefix('candles/') if endpoint.startswith('candles/') else ''
    if not (endpoint in {'summary', 'alarms', 'log'}
            or endpoint in {'book/' + key for key in BOOKS}
            or candle.isalnum() and len(candle) <= 16):
        raise HTTPException(404)
    query = query_string(endpoint, request.query_params)
    url = 'http://127.0.0.1:8996/api/' + quote(endpoint, safe='/')
    if query:
        url += '?' + query
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError('response too large')
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError('missing operational data')
        if endpoint.startswith('book/') and not isinstance(payload.get('equity'), (int, float)):
            raise ValueError('missing paper book')
    except (OSError, urllib.error.URLError, ValueError):
        raise HTTPException(503, '크립토 모의매매 서비스에 연결하지 못했습니다.')
    if endpoint in {'book/cnn', 'book/cnn_equal'}:
        payload['series'], payload['series_message'] = paper_series(endpoint.split('/')[1])
        payload['series_basis'] = SERIES_BASIS
    return payload
