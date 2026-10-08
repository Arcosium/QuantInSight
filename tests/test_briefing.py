import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from autofolio import briefing


def test_quotes_filter_invalid_data_and_keep_stale_cache(monkeypatch):
    briefing._CACHE.clear()
    clock = [1000.]
    monkeypatch.setattr(briefing.time, 'monotonic', lambda: clock[0])
    payload = {'chart': {'result': [{'timestamp': [100, 200, 300, 400],
        'indicators': {'quote': [{'close': [100, None, -1, 110]}]},
        'meta': {'regularMarketTime': 450}}]}}
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return json.dumps(payload).encode()
    def fetch(*args, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(briefing.urllib.request, 'urlopen', fetch)
    result = briefing.chart(briefing.MARKETS[0])
    assert result['points'] == [{'time': 100, 'close': 100.}, {'time': 400, 'close': 110.}]
    assert abs(result['change_pct'] - 10) < 1e-9
    assert result['as_of'] == briefing.iso(450)
    briefing.chart(briefing.MARKETS[0])
    assert len(calls) == 1 and calls[0]['timeout'] == 6
    clock[0] += 301
    def broken(*args, **kwargs): raise OSError('remote failure')
    monkeypatch.setattr(briefing.urllib.request, 'urlopen', broken)
    stale = briefing.chart(briefing.MARKETS[0])
    assert stale['status'] == 'stale' and stale['points'] == result['points']
    assert stale['as_of'] == result['as_of']
    assert briefing.chart(briefing.MARKETS[1])['status'] == 'unavailable'
    briefing._CACHE.clear()


def test_news_deduplicates_filters_dates_and_preserves_source(tmp_path, monkeypatch):
    path = tmp_path / 'news.db'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE articles(id INTEGER PRIMARY KEY,title,url,summary,source,source_label,published_at,collected_at)')
    entries = [
        ('비트코인 거래', 'https://example.com/a', '<b>거래량 증가</b>', '매체', '', '2026-10-08 10:00', '2026-10-08 10:05'),
        ('비트코인 거래', 'https://example.com/b', '중복', '매체', '', '2026-10-08 09:00', '2026-10-08 09:05'),
        ('미국 증시', 'https://example.com/c', '미국 증시', '매체', '', '', '2026-10-08 09:05'),
        ('오래된 기사', 'https://example.com/d', '지난 소식', '매체', '', '2026-01-01', '2026-10-08 09:05'),
        ('잘못된 링크', 'javascript:alert(1)', '', '매체', '', '2026-10-08', '2026-10-08'),
        ('ìëíìëêí', 'https://example.com/broken', '', '매체', '', '2026-10-08', '2026-10-08'),
        ('미래 기사', 'https://example.com/e', '', '매체', '', '2027-01-01', '2026-10-08'),
    ]
    db.executemany('INSERT INTO articles(title,url,summary,source,source_label,published_at,collected_at) VALUES(?,?,?,?,?,?,?)', entries)
    db.commit(); db.close()
    monkeypatch.setattr(briefing, 'NEWS_DB', path)
    result = briefing.news(datetime(2026, 10, 8, 3, tzinfo=timezone.utc))
    assert len(result['items']) == 2
    a, b = result['items']
    assert a['summary'] == '거래량 증가' and a['category'] == 'crypto'
    assert a['source'] == '매체' and a['published_at'].endswith('+09:00')
    assert b['summary'] == '' and b['time_basis'] == 'collected' and b['category'] == 'us'
    monkeypatch.setattr(briefing, 'NEWS_DB', tmp_path / 'missing')
    assert briefing.news()['status'] == 'unavailable'
    assert not (tmp_path / 'missing').exists()


def test_discovery_uses_completion_events_and_user_scope(tmp_path, monkeypatch):
    path = tmp_path / 'research.db'
    db = sqlite3.connect(path)
    db.executescript('CREATE TABLE events(kind,timestamp,body); CREATE TABLE alpha_candidates(id,user_id,status,result); CREATE TABLE strategies(id,payload,source);')
    now = 2_000_000.
    rows = [('good', 1, 2., 2., 2, now - 100, True, 'learned_v1'),
            ('private', 2, 2., 2., 2, now - 90, True, 'learned_v1'),
            ('old', 1, 2., 2., 2, now - 8 * 86400, True, 'learned_v1'),
            ('bad_period', 1, 2., 2., 2, now - 80, False, 'learned_v1'),
            ('weak', 1, 2., .5, 2, now - 70, True, 'learned_v1'),
            ('paper', 1, 2., 2., 2, now - 60, True, 'paper'),
            ('loss', 1, -2., 2., 2, now - 50, True, 'learned_v1')]
    for identity, owner, ret, sharpe, losses, stamp, accepted, engine in rows:
        payload = dict(id=identity,title=identity,owner_id=owner,market='crypto',net_return=ret,sharpe=sharpe,negative_months=losses,accepted=accepted,genome={'engine':engine})
        from autofolio.evaluation import PROTOCOL
        payload.update(evaluation_protocol=PROTOCOL,performance={'os':dict(months=9,net_return=ret,negative_months=losses,mdd=-.1,sharpe=sharpe)})
        db.execute('INSERT INTO strategies VALUES(?,?,?)', (identity,json.dumps(payload),identity))
        db.execute('INSERT INTO alpha_candidates VALUES(?,?,?,?)',(identity,owner,'done',identity))
        db.execute('INSERT INTO events VALUES(?,?,?)',('market_completed',stamp,json.dumps({'job':identity})))
    # A fresh catalogue refresh is not a new experiment completion.
    db.execute('INSERT INTO events VALUES(?,?,?)',('catalogue_refresh',now-1,json.dumps({'job':'old'})))
    db.commit();db.close()
    @contextmanager
    def connect():
        connection = sqlite3.connect(path);connection.row_factory=sqlite3.Row
        try: yield connection
        finally: connection.close()
    monkeypatch.setattr(briefing,'connect',connect)
    monkeypatch.setattr(briefing.period,'accepts',lambda value:value['accepted'])
    result = briefing.discoveries({'id':1},now)
    assert [r['id'] for r in result['items']] == ['good']
    assert result['items'][0]['discovered_at'] == briefing.iso(now-100)
    assert '통계적 유의성' in result['message']


def test_endpoint_requires_auth_before_external_io(monkeypatch):
    app = FastAPI();app.include_router(briefing.router)
    def unexpected(*args): raise AssertionError('Network must not run')
    monkeypatch.setattr(briefing, 'chart', unexpected)
    assert TestClient(app).get('/api/briefing').status_code == 401
