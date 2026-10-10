"""Read-only, explicitly non-ranking references for the shared experiment map."""
import copy
import json
import math
import os
import sqlite3
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from . import config
from .paper_benchmark import measure

ROOT = Path.home() / 'projects/HYFE_QTPA/work'
MODELS = {'cnn': '논문 · HeatF CNN 가중', 'cnn_equal': 'HeatF CNN 기본',
          'v1': 'GBM v1', 'v2': 'GBM v2', 'v2d': 'GBM 1d'}


def _read_json(path):
    if path.stat().st_size > 8_000_000:
        raise ValueError('Reference file too large')
    return json.loads(path.read_text())


def _positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _fold_returns(root, key):
    """Keep the original fold days, choosing the latest fold on overlap."""
    suffix = '_wev' if key == 'cnn' else ''
    folds = []
    for index in range(4):
        data = _read_json(root / f'ens_heatf10_direct_4h_s{index}_cohort_H84{suffix}.json')
        daily = sorted(data['daily'].items())
        if len(daily) < 2:
            raise ValueError('Missing fold NAV')
        for day, nav in daily:
            date.fromisoformat(day)
            if not _positive(nav):
                raise ValueError('Invalid fold NAV')
        folds.append((daily[0][0], {day.replace('-', ''): nav / daily[i-1][1] - 1
                                    for i, (day, nav) in enumerate(daily) if i}))
    returns = {}
    for _, values in sorted(folds):
        returns.update(values)
    return dict(sorted(returns.items()))


def _signal_returns(path):
    """Aggregate real recorded marks at UTC day end; never fill missing days."""
    if path.stat().st_size > 16_000_000:
        raise ValueError('Reference file too large')
    marks = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        ts, nav = row.get('ts'), row.get('equity')
        if not isinstance(ts, int) or isinstance(ts, bool) or not _positive(nav):
            raise ValueError('Invalid recorded NAV')
        marks[ts] = nav
    daily = {}
    for ts, nav in sorted(marks.items()):
        daily[datetime.fromtimestamp(ts / 1000, timezone.utc).strftime('%Y%m%d')] = nav
    values = sorted(daily.items())
    if len(values) < 2:
        raise ValueError('Insufficient recorded days')
    return {day: nav / values[i-1][1] - 1 for i, (day, nav) in enumerate(values) if i}


def _active_books(root):
    active = set()
    try:
        state = _read_json(root / 'live/state.json').get('dir', {})
        active.update(key for key in ('v1', 'v2', 'v2d') if state.get(key))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    database = Path(os.environ.get('HYFE_PAPER_DB', root / 'paper_trade/paper.sqlite3'))
    try:
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            row = db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        books = json.loads(row[0]).get('books', {}) if row else {}
        active.update(key for key, variant in (('cnn', 'event'), ('cnn_equal', 'equal'))
                      if _positive(books.get(variant, {}).get('net')))
    except (OSError, sqlite3.Error, ValueError, TypeError, AttributeError):
        pass
    return active


@lru_cache(maxsize=2)
def _crypto_points(bucket):
    active = _active_books(ROOT)
    points = []
    for key, title in MODELS.items():
        fold = key in ('cnn', 'cnn_equal')
        point = dict(id='reference-crypto-' + key, model_key=key, title=title, market='crypto',
                     reference=True, available=False, applied_targets=[], operational=key in active,
                     net_return=None, negative_months=None, months=None, start=None, end=None, sessions=0,
                     basis='논문 검증 fold · 일별 순자산' if fold else '페이퍼 장부 · 실현 순자산 · UTC 일 마감',
                     limitations=['최근 36개월 평가와 기간이 달라 순위·파레토 선발에서 제외합니다.',
                                  '롱숏 · 수수료 반영 · 펀딩비 제외',
                                  '검증 fold 사이의 공백을 채우지 않았습니다.' if fold else
                                  '미실현 손익 제외 · 초기 소급 장부와 이후 전진 운용 기록 포함'])
        try:
            returns = (_fold_returns(ROOT / 'results', key) if fold else
                       _signal_returns(ROOT / 'live' / f'dir_{key}_signals.jsonl'))
            point.update(available=True, **measure(returns), start=min(returns), end=max(returns), sessions=len(returns))
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            point['message'] = '원본 평가 기록을 확인할 수 없어 좌표를 표시하지 않습니다.'
        points.append(point)
    return points


def _timefolio_assignments(db,uid):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='timefolio_model_state'").fetchone():return []
    return db.execute('''SELECT a.strategy_id,a.target,s.payload FROM strategy_assignments a
        JOIN model_deployments d ON d.user_id=a.user_id AND d.target=a.target AND d.strategy_id=a.strategy_id AND d.status='ready'
        JOIN timefolio_model_state t ON t.user_id=a.user_id AND json_extract(t.body,'$.deployment')=d.id
        JOIN strategies s ON s.id=a.strategy_id
        WHERE a.user_id=? AND a.target='timefolio' AND a.status='timefolio_ready'
          AND json_extract(t.body,'$.connected')=1''',(uid,)).fetchall()


def _assignments(person, visible_rows):
    """An assigned ID alone is insufficient: require its current, ready paper book."""
    allowed = {row['id'] for row in visible_rows if row.get('owner_id') in (None, person['id'])}
    result = {}
    try:
        with sqlite3.connect(config.DB.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            rows = db.execute('''SELECT a.strategy_id,a.target FROM strategy_assignments a
                JOIN model_deployments d ON d.user_id=a.user_id AND d.target=a.target
                  AND d.strategy_id=a.strategy_id AND d.status='ready'
                JOIN paper_books p ON p.user_id=a.user_id AND p.deployment=d.id
                  AND a.target=p.market || '-paper'
                WHERE a.user_id=? AND a.status IN ('paper_ready','paper_active')
                  AND a.target IN ('kr-paper','us-paper','crypto-paper')''', (person['id'],)).fetchall()
            rows.extend((identity,target) for identity,target,_ in _timefolio_assignments(db,person['id']))
        for identity, target in rows:
            if identity in allowed and target not in result.setdefault(identity, []):
                result[identity].append(target)
    except (OSError, sqlite3.Error):
        pass
    return result


def _outside_points(market, person, visible_rows):
    """Keep an applied strategy visible after the rolling research period advances."""
    visible = {row['id'] for row in visible_rows}
    points = {}
    try:
        with sqlite3.connect(config.DB.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
            db.execute('PRAGMA query_only=ON')
            rows = db.execute('''SELECT a.strategy_id,a.target,s.payload FROM strategy_assignments a
                JOIN model_deployments d ON d.user_id=a.user_id AND d.target=a.target
                  AND d.strategy_id=a.strategy_id AND d.status='ready'
                JOIN paper_books p ON p.user_id=a.user_id AND p.deployment=d.id
                  AND a.target=p.market || '-paper'
                JOIN strategies s ON s.id=a.strategy_id
                WHERE a.user_id=? AND p.market=? AND a.status IN ('paper_ready','paper_active')
                  AND a.target IN ('kr-paper','us-paper','crypto-paper')''', (person['id'], market)).fetchall()
            if market=='timefolio':rows.extend(_timefolio_assignments(db,person['id']))
        for identity, target, payload in rows:
            if identity in visible:
                continue
            row = json.loads(payload)
            if row.get('owner_id') not in (None, person['id']) or row.get('market', 'timefolio') != market:
                continue
            if identity in points:
                if target not in points[identity]['applied_targets']:
                    points[identity]['applied_targets'].append(target)
                continue
            good = all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                       and math.isfinite(row[k]) for k in ('net_return', 'negative_months', 'months'))
            good = good and row['negative_months'] >= 0 and row['months'] > 0
            for key in ('start', 'end'):
                value = str(row.get(key, ''))
                if len(value) != 8 or not value.isdigit():
                    good = False
                else:
                    try:
                        datetime.strptime(value, '%Y%m%d')
                    except ValueError:
                        good = False
            point = dict(id='reference-assigned-' + identity, strategy_id=identity,
                         title=row.get('title') or '운용 중 전략', market=market, reference=True,
                         available=bool(good), applied_targets=[target],
                         basis='현재 적용 전략 · 적용 당시 백테스트',
                         limitations=['현재 연구소 평가 기간과 달라 순위·파레토 선발에서 제외합니다.',
                                      '좌표는 적용 당시 백테스트이며 실제 계좌 수익률이 아닙니다.'],
                         **{k: row.get(k) if good else None for k in
                            ('net_return', 'negative_months', 'months', 'start', 'end', 'sessions')})
            if not good:
                point['message'] = '적용 중이지만 원본 평가 좌표를 확인할 수 없습니다.'
            points[identity] = point
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        pass
    return list(points.values())


def references(market, person, visible_rows):
    """Return reference points separately so callers cannot rank or evolve them."""
    assignments = _assignments(person, visible_rows)
    points = copy.deepcopy(_crypto_points(int(time.time() // 60))) if market == 'crypto' else []
    for point in points:
        if person.get('role') == 'admin' and point.pop('operational', False):
            point['applied_targets'] = ['crypto-paper']
        else:
            point.pop('operational', None)
    points.extend(_outside_points(market, person, visible_rows))
    return dict(points=points, assignments=assignments)
