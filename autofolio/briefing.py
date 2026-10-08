"""Read-only market brief: public quotes, local news and scoped research findings."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from .auth import user
from . import period, research
from .store import connect
from .evaluation import selection_metrics

router = APIRouter()
KST = ZoneInfo('Asia/Seoul')
NEWS_DB = Path.home() / 'projects/lib/data/arcnews.db'
MARKETS = (('kospi', '코스피', '^KS11', 'KRW'),
           ('sp500', 'S&P 500', '^GSPC', 'USD'),
           ('btc', 'BTC', 'BTC-USD', 'USD'))
_CACHE = {}
_LOCKS = {key: threading.Lock() for key, *_ in MARKETS}
CRITERIA = '최근 7일 완료 · OS 9개월 수익률 > 0 · 샤프 ≥ 1 · 손실월 ≤ 3'


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _quote(spec):
    key, label, symbol, currency = spec
    base = dict(id=key, label=label, symbol=symbol, currency=currency,
                source='Yahoo Finance', source_url='https://finance.yahoo.com/quote/' + urllib.parse.quote(symbol, safe=''),
                interval='1d', range='3mo', points=[], latest=None, change_pct=None, as_of=None)
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol, safe='') + '?range=3mo&interval=1d'
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 QuantInSight/1.0'})
    with urllib.request.urlopen(request, timeout=6) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Oversized quote response')
    result = json.loads(raw)['chart']['result'][0]
    closes = result['indicators']['quote'][0]['close']
    points = sorted({int(t): float(c) for t, c in zip(result['timestamp'], closes)
                     if finite(t) and finite(c) and c > 0}.items())[-100:]
    if len(points) < 2:
        raise ValueError('Insufficient quote observations')
    meta = result.get('meta', {})
    stamp = meta.get('regularMarketTime')
    stamp = stamp if finite(stamp) and stamp >= points[-1][0] else points[-1][0]
    return dict(base, points=[dict(time=t, close=c) for t, c in points], latest=points[-1][1],
                change_pct=(points[-1][1] / points[-2][1] - 1) * 100,
                as_of=iso(stamp), fetched_at=iso(time.time()), status='ok',
                message='최근 3개월 일봉 · 직전 관측 종가 대비 · 제공 시세는 지연될 수 있음')


def chart(spec):
    """Bound requests per symbol; failures preserve an explicitly stale last success."""
    key, label, symbol, currency = spec
    with _LOCKS[key]:
        now = time.monotonic()
        previous = _CACHE.get(key)
        if previous and now < previous[0]:
            return previous[1]
        try:
            value = _quote(spec)
            ttl = 300
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            value = dict(previous[1]) if previous else dict(
                id=key, label=label, symbol=symbol, currency=currency, source='Yahoo Finance',
                source_url='https://finance.yahoo.com/quote/' + urllib.parse.quote(symbol, safe=''),
                interval='1d', range='3mo', points=[], latest=None, change_pct=None, as_of=None)
            value.update(status='stale' if value['points'] else 'unavailable',
                         message='시세 연결 지연 · 마지막 수신 자료' if value['points'] else '시세를 불러오지 못했습니다.')
            ttl = 30
        _CACHE[key] = (time.monotonic() + ttl, value)
        return value


def _date(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        try:
            result = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    # The Korean public news collector supplies local wall-clock strings.
    return result if result.tzinfo else result.replace(tzinfo=KST)


def _plain(value, limit):
    return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]*>', '', value or ''))).strip()[:limit]


def _category(title):
    if re.search(r'비트코인|암호화폐|가상자산|크립토|이더리움|블록체인|\bBTC\b', title, re.I):
        return 'crypto'
    if re.search(r'미국|뉴욕|나스닥|월가|연준|S&P|다우|트럼프|엔비디아', title, re.I):
        return 'us'
    if re.search(r'코스피|코스닥|국내|한국|금융|증시|주식|증권|삼성|하이닉스|원화', title):
        return 'kr'
    return 'world'


def news(now=None):
    now = now or datetime.now(timezone.utc)
    items, seen = [], set()
    try:
        db = sqlite3.connect(f'file:{NEWS_DB}?mode=ro', uri=True, timeout=2)
        db.row_factory = sqlite3.Row
        try:
            found = db.execute('SELECT title,url,summary,source,source_label,published_at,collected_at '
                               'FROM articles ORDER BY collected_at DESC,id DESC LIMIT 1000').fetchall()
        finally:
            db.close()
        for row in found:
            title = _plain(row['title'], 180)
            if title.count('�') > 1 or (len(re.findall(r'[À-ÿ]', title)) > 4 and len(re.findall(r'[가-힣]', title)) < 5):
                continue
            published = _date(row['published_at'])
            collected = _date(row['collected_at'])
            stamp = published or collected
            url = row['url'] or ''
            if (not title or urllib.parse.urlsplit(url).scheme not in ('http', 'https') or not stamp
                    or not now - timedelta(days=7) <= stamp <= now + timedelta(minutes=10)):
                continue
            identity = re.sub(r'[^\w]', '', title).lower()
            if identity in seen or url in seen:
                continue
            seen.update((identity, url))
            description = _plain(row['summary'], 220)
            if description == title:
                description = ''
            items.append(dict(title=title, summary=description, source=row['source'] or row['source_label'],
                              url=url, published_at=published.isoformat() if published else None,
                              collected_at=collected.isoformat() if collected else None,
                              category=_category(title), time_basis='published' if published else 'collected'))
        items.sort(key=lambda r: _date(r['published_at'] or r['collected_at']), reverse=True)
        # Preserve coverage without letting one topic consume the brief.
        selected, counts = [], {}
        for item in items:
            category = item['category']
            if counts.get(category, 0) >= 4:
                continue
            selected.append(item)
            counts[category] = counts.get(category, 0) + 1
            if len(selected) == 12:
                break
        return dict(items=selected, status='ok' if selected else 'empty',
                    message='최근 7일 · 수집된 기사 제목과 제공 설명 · 제목 기준 시장 분류' if selected else '최근 7일 내 수집된 기사가 없습니다.')
    except (sqlite3.Error, OSError):
        return dict(items=[], status='unavailable', message='뉴스 수집 자료를 불러오지 못했습니다.')


def discoveries(person, now=None):
    now = now if now is not None else time.time()
    items = []
    try:
        with connect() as db:
            completed = db.execute("""
                SELECT s.payload, MAX(e.timestamp) completed
                FROM events e JOIN alpha_candidates a
                  ON a.id=json_extract(CASE WHEN json_valid(e.body) THEN e.body ELSE '{}' END,'$.job')
                JOIN strategies s ON s.source=a.result
                WHERE e.kind='market_completed' AND e.timestamp BETWEEN ? AND ?
                  AND a.status='done' AND a.user_id=?
                GROUP BY s.id ORDER BY completed DESC LIMIT 512
            """, (now - 7 * 86400, now, person['id'])).fetchall()
        for record in completed:
            value = json.loads(record['payload'])
            if not research.visible(value, person) or not period.accepts(value):
                continue
            if (value.get('genome') or {}).get('engine') != 'learned_v1':
                continue
            metric=selection_metrics(value)
            if metric is None:continue
            value=dict(value,**{k:metric.get(k) for k in ('net_return','sharpe','negative_months')})
            if not all(finite(value.get(k)) for k in ('net_return', 'sharpe', 'negative_months')):
                continue
            if value['net_return'] <= 0 or value['sharpe'] < 1 or value['negative_months'] > 3:
                continue
            items.append(dict((key, value.get(key)) for key in ('id', 'title', 'market', 'net_return', 'sharpe', 'negative_months')) |
                         dict(discovered_at=iso(record['completed'])))
            if len(items) == 8:
                break
    except (sqlite3.Error, ValueError, TypeError):
        return dict(items=[], status='unavailable', criteria=CRITERIA, message='연구 완료 기록을 불러오지 못했습니다.')
    return dict(items=items, status='ok' if items else 'empty', criteria=CRITERIA,
                message='탐색 후보 선별 기준이며 통계적 유의성이나 독립 검증을 뜻하지 않습니다.')


@router.get('/api/briefing')
def briefing(request: Request):
    person = user(request)
    with ThreadPoolExecutor(max_workers=3) as pool:
        charts = list(pool.map(chart, MARKETS))
    return dict(as_of=iso(time.time()), charts=charts, news=news(), discoveries=discoveries(person))
