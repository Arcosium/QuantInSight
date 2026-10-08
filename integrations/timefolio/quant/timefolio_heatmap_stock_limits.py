"""Dated Timefolio position limits and an independent pre-July Hynix audit.

Official notice 708 (2026-06-16) changes Hynix from 15% to 30% on 2026-07-01.
Samsung remains 40%; other individual limits are 15%. No account integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import DEST as BASE_ROOT, amend_actions
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE

DEST = SOURCE.with_name('20260929_dated_limits_audit_v1')
PRIOR_NAMES = ['20260928_action_amendment_v5', '20260928_seed_evaluation_v1',
               '20260928_portfolio_selector_v1', '20260928_allocation_probe_v1',
               '20260928_allocation_expansion_v2', '20260928_longonly_evaluation_v1',
               '20260929_snapshot_v2', '20260929_consensus_v1']


def historical_stock_caps(codes, dates):
    limits = np.full((len(codes), len(dates)), .15)
    if '005930' in codes: limits[codes.index('005930')] = .40
    if '000660' in codes:
        limits[codes.index('000660'), np.asarray(dates) >= '20260701'] = .30
    return limits


def audit_pre_july_hynix(p, ix, result):
    """Rebuild NAV after every buy; a buy in any other name also shrinks NAV."""
    codes = {code: i for i, code in enumerate(ix['codes'])}
    if '000660' not in codes:
        return {'post_buy_limit_errors': [], 'closing_breaches': [], 'maximum_weight': 0., 'buy_checks': 0}
    dates = {date: d for d, date in enumerate(ix['dates'])}; h = codes['000660']
    events = {}
    for trade in result['trades']: events.setdefault(trade['date'], []).append(trade)
    qty = np.zeros(len(codes), dtype=np.int64); marks = np.zeros(len(codes)); cash = 1e9
    errors = []; closing = []; maximum = 0.; checks = 0
    for row in result['daily']:
        date = row['date']
        if date >= '20260701': break
        d = dates[date]; ratio = p['split'][:, d]
        qty = np.floor(qty*ratio+1e-7).astype(np.int64)
        marks = np.divide(marks, ratio, out=marks.copy(), where=ratio > 0)
        opening = p['exec_price'][:, d].astype(float)
        good = np.isfinite(opening) & (opening > 0) & (p['exec_count'][:, d] >= 25)
        marks[good] = opening[good]
        for trade in events.get(date, []):
            i = codes[trade['code']]; buy = trade['side'] == 'buy'; q = trade['qty']
            qty[i] += q if buy else -q
            cash += (-q*trade['price'] if buy else q*trade['price'])-trade['fee']
            if buy:
                checks += 1; nav = cash+qty@marks; weight = float(qty[h]*marks[h]/nav)
                maximum = max(maximum, weight)
                if qty[h]*marks[h] > .15*nav+1:
                    errors.append({'date': date, 'bought': trade['code'], 'weight': weight, 'limit': .15})
        known = np.isfinite(p['close'][:, d]) & (p['close'][:, d] > 0)
        marks[known] = p['close'][known, d]; nav = cash+qty@marks
        maximum = max(maximum, float(qty[h]*marks[h]/nav))
        if qty[h]*marks[h] > .15*nav+1:
            closing.append({'date': date, 'weight': float(qty[h]*marks[h]/nav), 'limit': .15})
        if abs(nav-row['nav']) > .01: raise AssertionError('Dated-cap audit NAV reconstruction differs')
    return {'post_buy_limit_errors': errors, 'closing_breaches': closing, 'maximum_weight': maximum, 'buy_checks': checks}


def run_prior_audit():
    DEST.mkdir(exist_ok=True); base = json.loads((BASE_ROOT/'protocol.json').read_text())
    p, ix, _, _ = context(SOURCE); p, _ = amend_actions(p, ix, base['actions'])
    files = [Path(__file__), Path(__file__).with_name('timefolio_heatmap_dated_replay.py'),
             SOURCE.with_name('20260928_planning_probe_v1')/'official_rules/708.json']
    spec = {'source': 'Timefolio official notice708, 2026-06-16; execution-date transition20260701',
            'roots': PRIOR_NAMES, 'audit': 'all stored portfolio paths in these eight final ledger roots; independently reconstruct pre-July NAV and inspect Hynix after every buy and close',
            'hashes': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    target = DEST/'protocol.json'
    if target.exists() and json.loads(target.read_text()) != spec: raise RuntimeError('Audit protocol changed')
    if not target.exists(): atomic_json(target, spec)
    outputs = {}; maximum = 0.; examined = 0; skipped = 0; failed = []; drift = []
    for name in PRIOR_NAMES:
        for path in sorted((SOURCE.with_name(name)/'portfolios').glob('*.json')):
            result = json.loads(path.read_text()); key = name+'/'+path.stem
            if not any(t['code'] == '000660' and t['date'] < '20260701' for t in result['trades']):
                skipped += 1; outputs[key] = {'no_pre_july_hynix_trade': True}; continue
            check = audit_pre_july_hynix(p, ix, result); outputs[key] = check; examined += 1
            maximum = max(maximum, check['maximum_weight'])
            if check['post_buy_limit_errors']: failed.append(key)
            if check['closing_breaches']: drift.append(key)
        print(json.dumps({'root': name, 'portfolios': len(outputs)}), flush=True)
    atomic_json(DEST/'per_portfolio.json', outputs)
    summary = {'portfolios': len(outputs), 'reconstructed': examined, 'no_pre_july_hynix_trade': skipped,
               'maximum_observed_pre_july_hynix_weight': maximum, 'post_buy_failed_paths': failed,
               'closing_breach_paths': drift, 'no_other_rules_certified_by_this_audit': True}
    atomic_json(DEST/'summary.json', summary); print(json.dumps(summary), flush=True)
    if failed: raise AssertionError('Previous portfolios need dated-cap replay before new inference')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['audit-prior']); ap.parse_args()
    run_prior_audit()
