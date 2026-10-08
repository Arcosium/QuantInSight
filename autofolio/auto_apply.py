"""Opt-in, user-scoped automatic paper deployment of newly observed Pareto models."""
import contextlib
import fcntl
import json
import math
import time
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from . import store, research
from .auth import user, mutation_guard
from .catalogue import rows
from .period import accepts
from .metrics import pareto_front
from .evaluation import selection_metrics

router = APIRouter()
TARGETS = {'kr': 'kr-paper', 'us': 'us-paper', 'crypto': 'crypto-paper', 'timefolio': 'timefolio'}
LABELS = {'kr': '한국주식 페이퍼매매', 'us': '미국주식 페이퍼매매', 'crypto': '크립토 페이퍼매매'}


def initialize():
    with store.connect() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS auto_apply_policies(
            user_id INTEGER NOT NULL,market TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 0,
            revision INTEGER NOT NULL DEFAULT 0,seen TEXT NOT NULL DEFAULT '[]',
            updated REAL NOT NULL,message TEXT NOT NULL DEFAULT '',PRIMARY KEY(user_id,market));
            CREATE TABLE IF NOT EXISTS auto_apply_jobs(
            deployment_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,market TEXT NOT NULL,revision INTEGER NOT NULL);''')


@contextlib.contextmanager
def lock():
    store.DATA.mkdir(parents=True, exist_ok=True)
    with (store.DATA / 'auto-apply.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def eligible(person, market, candidates=None):
    selected=[]
    for r in (rows() if candidates is None else candidates):
        if not research.visible(r,person,market) or not accepts(r):continue
        if (r.get('genome') or {}).get('engine')!='learned_v1':continue
        metrics=selection_metrics(r)
        if metrics is not None:
            selected.append(dict(r,net_return=metrics['net_return'],negative_months=metrics['negative_months']))
    return selected


def state(uid, market):
    if market not in TARGETS:
        raise HTTPException(404)
    initialize()
    with store.connect() as db:
        policy = db.execute('SELECT * FROM auto_apply_policies WHERE user_id=? AND market=?', (uid, market)).fetchone()
    available = market != 'timefolio'
    message = (f'{LABELS.get(market)} · 켠 이후의 새 OS 9개월 파레토 후보 중 수익률 최대, 동률이면 손실월 최소 · 재학습 검증 후 전환'
               if available else '타임폴리오 13회 운용 모델 연결·대회 규칙 검증 전입니다. 자동 적용은 잠겨 있습니다.')
    if market == 'kr':
        message += ' · 한투 실매매 정지 유지'
    return dict(enabled=bool(policy and policy['enabled']), available=available, target=TARGETS[market],
                message=message, last_action=policy['message'] if policy else '', updated=policy['updated'] if policy else None)


class Toggle(BaseModel):
    enabled: bool


@router.get('/api/research/{market}/auto-apply')
def get_policy(market: str, request: Request):
    return state(user(request)['id'], market)


@router.post('/api/research/{market}/auto-apply')
def set_policy(market: str, value: Toggle, request: Request):
    mutation_guard(request)
    return configure(user(request), market, value.enabled)


def configure(person, market, enabled):
    current = state(person['id'], market)
    if enabled and not current['available']:
        raise HTTPException(409, current['message'])
    with lock():
        # A repeated enabled request must not erase the observation watermark.
        current = state(person['id'], market)
        if current['enabled'] == enabled:
            return current
        seen = json.dumps([r['id'] for r in eligible(person, market)])
        with store.connect() as db:
            db.execute('''INSERT INTO auto_apply_policies VALUES(?,?,?,1,?,?,?)
                ON CONFLICT(user_id,market) DO UPDATE SET enabled=excluded.enabled,
                revision=auto_apply_policies.revision+1,seen=excluded.seen,updated=excluded.updated,message=excluded.message''',
                (person['id'], market, int(enabled), seen, time.time(), '새 파레토 후보 대기' if enabled else '자동 적용 중지'))
            if not enabled:
                db.execute("""UPDATE model_deployments SET status='cancelled',message='자동 적용 해제',updated=?
                    WHERE status='queued' AND id IN (SELECT deployment_id FROM auto_apply_jobs WHERE user_id=? AND market=?)""",
                    (time.time(), person['id'], market))
    return state(person['id'], market)


@contextlib.contextmanager
def activation(identity):
    """Hold the same cross-process lock as disable through the deployment commit."""
    initialize()
    with lock():
        with store.connect() as db:
            row = db.execute('''SELECT j.revision job_revision,p.revision,p.enabled FROM auto_apply_jobs j
                LEFT JOIN auto_apply_policies p ON p.user_id=j.user_id AND p.market=j.market
                WHERE j.deployment_id=?''', (identity,)).fetchone()
        if row and (not row['enabled'] or row['revision'] != row['job_revision']):
            raise ValueError('자동 적용 해제로 모델 전환을 취소했습니다.')
        yield


def tick():
    initialize()
    from .auth import connect as auth_connect
    from .deployment import request_retrain, initialize as deployment_initialize
    deployment_initialize()
    with lock():
        with store.connect() as db:
            policies = [dict(p) for p in db.execute('SELECT * FROM auto_apply_policies WHERE enabled=1')]
        if not policies:
            return
        with auth_connect() as db:
            people = {r['id']: dict(r) for r in db.execute('SELECT id,role FROM users')}
        all_rows = rows()
        for policy in policies:
            uid, market = policy['user_id'], policy['market']
            if uid not in people or market not in LABELS:
                continue
            candidates = eligible(people[uid], market, all_rows)
            seen = set(json.loads(policy['seen']))
            with store.connect() as db:
                busy = db.execute("SELECT 1 FROM model_deployments WHERE user_id=? AND target=? AND status IN ('queued','training')", (uid, TARGETS[market])).fetchone()
            if busy:
                continue
            # Do not compare different session calendars as one Pareto population.
            front = []
            for cohort in {r['cohort'] for r in candidates}:
                front.extend(pareto_front([r for r in candidates if r['cohort'] == cohort]))
            fresh = sorted((r for r in front if r['id'] not in seen),
                           key=lambda r: (-r['net_return'], r['negative_months'], r['id']))
            message = policy['message']
            if fresh:
                chosen = fresh[0]
                try:
                    request_retrain(uid, chosen['id'], TARGETS[market], auto_policy=(market, policy['revision']))
                    message = f"새 파레토 전략 재학습 요청 · {chosen['title']}"
                except ValueError as exc:
                    message = f'자동 적용 보류 · {exc}'[:250]
                store.event('auto_apply', dict(user_id=uid, market=market, strategy=chosen['id'], message=message))
            with store.connect() as db:
                db.execute('UPDATE auto_apply_policies SET seen=?,updated=?,message=? WHERE user_id=? AND market=?',
                           (json.dumps([r['id'] for r in candidates]), time.time(), message, uid, market))
