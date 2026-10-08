"""Finite, fair development experiments from the academic chart screen.

KR's saved CNN was strong in B/C/D but lost in A; US results were unstable.
Start with cheap price baselines, then chart/regime/holding-period controls.
This is adaptive development infrastructure, never independent OOS evidence.
The result source is 2026-10-08_chart_vision_screen_v1/CHART_IMAGE_SCREEN_RESULTS.json.
"""
import hashlib
import itertools
import json

from . import research, store

PROVIDER = 'study-campaign-v1'
STATE_KEY = 'research_campaign_state'
MARKETS = ('kr', 'us')
PENDING = ('queued', 'running', 'waiting_data')
ORDER = ('momentum', 'reversal', 'trend_momentum', 'lowvol_momentum',
         'gasf', 'mtf', 'image_rank_ensemble', 'fourier', 'candle_hog', 'recurrence')
BASELINES = {'momentum', 'reversal', 'trend_momentum', 'lowvol_momentum'}


def _identity(uid, market, genome, period):
    # Must match research.save_candidates, including its JSON separators.
    body = json.dumps(genome, sort_keys=True)
    return hashlib.sha256(f'{uid}:{market}:{period}:{body}'.encode()).hexdigest()[:20]


def proposals():
    """Yield baseline comparison, single-variable checks, then remaining space."""
    domain = research.STOCK_DOMAINS
    preferred = dict(selection_count=20, holding_sessions=14, ridge_alpha=10.,
                     mode='rank_only', window=20)
    base = {k: preferred.get(k, values[0]) if preferred.get(k) in values else values[0]
            for k, values in domain.items()}
    representations = [r for r in ORDER if r in domain['representation']]
    representations += [r for r in domain['representation'] if r not in representations]
    seen = set()

    def admissible(genome):
        # Price baselines have no fitted Ridge model, so alpha variations are duplicates.
        return genome['representation'] not in BASELINES or genome['ridge_alpha'] == base['ridge_alpha']

    def emit(genome, phase):
        key = json.dumps(genome, sort_keys=True)
        if key not in seen and admissible(genome):
            seen.add(key)
            return genome, phase
        return None

    centers = [dict(base, representation=r) for r in representations]
    for genome in centers:
        item = emit(genome, 'baseline_comparison')
        if item: yield item
    # Holding changes test persistence/turnover; regime gating tests the weak A regime.
    for field in ('mode', 'holding_sessions', 'selection_count', 'ridge_alpha', 'window'):
        for genome in centers:
            for value in domain[field]:
                item = emit(dict(genome, **{field: value}), 'single_variable_checks')
                if item: yield item
    keys = list(domain)
    for values in itertools.product(*(domain[k] for k in keys)):
        item = emit(dict(zip(keys, values)), 'combined_checks')
        if item: yield item


def _rows(uid):
    research.initialize()
    with store.connect() as db:
        return [dict(r) for r in db.execute(
            "SELECT id,market,status,definition,provider FROM alpha_candidates WHERE user_id=? AND market IN ('kr','us')",
            (uid,)).fetchall()]


def replenish(uid, limit=4):
    """Keep at most `limit` stock jobs pending, including manually seeded jobs.

    Runs on the single worker tick. No keys, model calls, retries of failed
    definitions, or unbounded database materialization. New month = new cohort.
    """
    if not store.setting('research_campaign_enabled', False):
        return []
    limit = max(0, min(int(limit), 32))
    rows = _rows(uid)
    period = research.window()
    # Retire only this campaign's unstarted previous-window jobs. Running or
    # manually requested evaluations are never cancelled by queue maintenance.
    stale = [r['id'] for r in rows if r['provider'] == PROVIDER
             and r['status'] in ('queued', 'waiting_data')
             and r['id'] != _identity(uid, r['market'], json.loads(r['definition']), period)]
    if stale:
        with store.connect() as db:
            db.executemany("UPDATE alpha_candidates SET status='retired',message='평가 기간 갱신으로 대기 실험 종료' WHERE id=? AND status IN ('queued','waiting_data') AND provider=?",
                           [(identity, PROVIDER) for identity in stale])
        rows = _rows(uid)
    available = max(0, limit - sum(r['status'] in PENDING for r in rows))
    if not available:
        return []
    known = {r['id'] for r in rows}
    state = store.setting(STATE_KEY, {})
    next_market = state.get('next_market', 'kr')
    if next_market not in MARKETS: next_market = 'kr'
    plans = list(proposals())
    ready = {market: research.protocol(market)['ready'] for market in MARKETS}
    waiting = {market: sum(r['market'] == market and r['status'] == 'waiting_data'
                           for r in rows) for market in MARKETS}
    added = []
    phase = 'exhausted'
    while len(added) < available:
        chosen = None
        # Alternate markets even across restarts and limit=1; allow the other if exhausted.
        for market in (next_market, MARKETS[1 - MARKETS.index(next_market)]):
            if not ready[market] and (waiting[market] >= 1
                    or (available - len(added) <= 1 and any(ready.values()))):
                continue
            for genome, candidate_phase in plans:
                identity = _identity(uid, market, genome, period)
                if identity not in known:
                    chosen = market, genome, candidate_phase, identity
                    break
            if chosen: break
        if not chosen: break
        market, genome, phase, identity = chosen
        known.add(identity)
        saved = research.save_candidates(uid, market, [genome], provider=PROVIDER)
        added.extend(saved)
        if saved and not ready[market]: waiting[market] += len(saved)
        next_market = MARKETS[1 - MARKETS.index(market)]
    remaining = {market: sum(_identity(uid, market, genome, period) not in known
                             for genome, _ in plans) for market in MARKETS}
    store.set_setting(STATE_KEY, dict(user_id=uid, period=list(period), next_market=next_market,
                                     phase=phase, last_added=len(added), remaining=remaining))
    return added


def status(uid=None):
    state = store.setting(STATE_KEY, {})
    uid = uid if uid is not None else state.get('user_id')
    enabled = bool(store.setting('research_campaign_enabled', False))
    if uid is None:
        return dict(enabled=enabled, phase='not_started', pending=0, done=0, failed=0, remaining=0,
                    markets={})
    period = research.window()
    rows = _rows(uid)
    known = {r['id']: r for r in rows}
    plan = list(proposals())
    markets = {}
    phases = []
    for market in MARKETS:
        ids = {_identity(uid, market, genome, period): phase for genome, phase in plan}
        current = [known[i] for i in ids if i in known]
        unseen = [phase for identity, phase in ids.items() if identity not in known]
        phases.extend(unseen[:1])
        markets[market] = dict(total=len(ids), remaining=len(unseen),
            pending=sum(r['status'] in PENDING for r in current),
            done=sum(r['status'] == 'done' for r in current),
            failed=sum(r['status'] == 'failed' for r in current))
    remaining = sum(m['remaining'] for m in markets.values())
    pending = sum(r['status'] in PENDING for r in rows)
    phase_order = ('baseline_comparison', 'single_variable_checks', 'combined_checks')
    phase = min(phases, key=phase_order.index) if phases else 'draining' if pending else 'exhausted'
    return dict(enabled=enabled, period=list(period), phase=phase if enabled else 'disabled',
                pending=pending, done=sum(m['done'] for m in markets.values()),
                failed=sum(m['failed'] for m in markets.values()), remaining=remaining,
                markets=markets, evidence='development_only')
