"""Isolated public KIND issuance details, preserving unknown security classes.

Five-character issuer IDs select acquisition candidates only. They are never
converted into six-character stock codes or treated as a share-class mapping.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import requests

from quant.timefolio_cnn_history import sha, write_json

URL = 'https://kind.krx.co.kr/corpgeneral/stockissuelist.do'
LABELS = {'회사명': 'company_name', '상장일': 'listing_date', '상장방식': 'listing_type',
          '발행사유': 'reason', '발행일': 'issue_date', '액면가': 'face_value',
          '발행가': 'issue_price', '발행주식수': 'share_change', '누적발행주식수': 'cumulative_issued_shares'}


def parse_detail(html, event):
    soup = BeautifulSoup(html, 'html.parser'); out = {}
    for th in soup.find_all('th'):
        label = th.get_text('', strip=True).replace(' ', '')
        cell = th.find_next_sibling('td')
        if label not in LABELS or cell is None:
            continue
        key = LABELS[label]; value = cell.get_text(' ', strip=True)
        if key in out and out[key] != value:
            raise ValueError('Conflicting KIND detail fields')
        out[key] = value
    for key in ['company_name', 'listing_date', 'listing_type', 'share_change', 'cumulative_issued_shares']:
        if not out.get(key):
            raise ValueError('Missing KIND issuance detail field: '+key)
    for key in ['share_change', 'cumulative_issued_shares', 'face_value', 'issue_price']:
        raw = out.get(key, '').replace(',', '')
        if key in ['share_change', 'cumulative_issued_shares'] and not re.fullmatch(r'-?\d+', raw):
            raise ValueError('Noninteger share quantity')
        out[key] = int(raw) if re.fullmatch(r'-?\d+', raw) else None
    for key in ['listing_date', 'issue_date']:
        raw = out.get(key, '')
        out[key] = datetime.strptime(raw, '%Y-%m-%d').strftime('%Y%m%d') if raw else None
    if (out['listing_date'] != event['date'] or out['listing_type'] != event['listing_type']
            or out['share_change'] != event['shares_signed']):
        raise ValueError('KIND list and detail disagree')
    if re.sub(r'\s+', '', out['company_name']) != re.sub(r'\s+', '', event['company_name']):
        raise ValueError('KIND issuer name mismatch')
    if out['cumulative_issued_shares'] < 0 or out['cumulative_issued_shares']-out['share_change'] < 0:
        raise ValueError('Negative outstanding share count')
    out.update(event_id=event['event_id'], issuer_popup_id=event['issuer_popup_id'],
        implied_previous_issued_shares=out['cumulative_issued_shares']-out['share_change'],
        security_code=None, share_class=None, share_class_verified=False,
        publication_time_verified=False, issued_equals_listed_verified=False,
        source=URL, detail_method=event['detail_method'])
    return out


def match_share_chains(records, baseline):
    """Audit arithmetic continuity; do not claim official class identification.

    Multiple same-day changes can arrive out of order. Only a unique transition
    from a unique baseline class is assigned. Unknown transitions remain visible
    and make later common-share history uncertified instead of being ignored.
    """
    state = {name: int(value) for name, value in baseline.items()}
    if not state or any(value <= 0 for value in state.values()):
        raise ValueError('Positive share-class baselines required')
    identities = [(r['event_id'], r['issuer_popup_id']) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError('Deduplicate exact source occurrences before ledger application')
    applied = []; unresolved = []
    for day in sorted({r['listing_date'] for r in records}):
        pending = [dict(r) for r in records if r['listing_date'] == day]
        while pending:
            choices = []
            for i, row in enumerate(pending):
                matches = [name for name, value in state.items() if value == row['implied_previous_issued_shares']]
                if len(matches) == 1:
                    choices.append((i, matches[0]))
            counts = {name: sum(label == name for _, label in choices) for name in state}
            unique = [(i, name) for i, name in choices if counts[name] == 1]
            if not unique:
                unresolved.extend(pending)
                break
            # Different classes commute; stable input order is retained for audit.
            i, name = unique[0]; row = pending.pop(i)
            state[name] = row['cumulative_issued_shares']
            applied.append(dict(event_id=row['event_id'], listing_date=day,
                inferred_share_class=name, previous=row['implied_previous_issued_shares'],
                current=row['cumulative_issued_shares'], delta=row['share_change'],
                official_share_class_verified=False))
    return dict(final_counts=state, applied=applied, unresolved=unresolved,
                complete_arithmetic_chain=not unresolved, historical_listed_shares_certified=False)


def run(groups_path, cohort_path, output, *, probe_cache=None, interval=1.):
    if interval < 1:
        raise ValueError('Public KIND acquisition is limited to one request per second')
    groups_path, cohort_path, output = map(Path, [groups_path, cohort_path, output])
    groups = json.loads(groups_path.read_text()); cohort = json.loads(cohort_path.read_text())
    candidates = {r['code'][:5] for r in cohort['receipts']}
    if any(g['conflicting'] for g in groups):
        raise ValueError('Resolve conflicting event identities first')
    events = [g['canonical_record'] for g in groups
              if g['canonical_record']['issuer_popup_id'] in candidates
              and g['canonical_record']['detail_method'] == 'searchStockIssueDetail']
    priority = {name: rank for rank, name in enumerate(['00593', '00066', '03572', '03542', '00018', '00004'])}
    events.sort(key=lambda e: (priority.get(e['issuer_popup_id'], 100), e['issuer_popup_id'], e['date'], e['event_id']))
    plan = dict(groups_sha256=sha(groups_path), cohort_sha256=sha(cohort_path), source_sha256=sha(__file__),
        source=URL, interval_seconds=interval, events=events,
        candidate_selection='First five code characters only prioritize acquisition; they DO NOT identify the security or class.')
    output.mkdir(parents=True, exist_ok=True); plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen KIND detail plan changed')
    if not plan_path.exists():
        write_json(plan_path, plan)
    session = requests.Session(); began = time.monotonic(); last_request = 0.; records = []
    try:
        for i, event in enumerate(events):
            identity = event['issuer_popup_id']+'_'+event['event_id']
            destination = output/(identity+'.json'); html_path = output/(identity+'.html')
            receipt = output/(identity+'.receipt.json')
            if receipt.exists():
                proof = json.loads(receipt.read_text())
                if (proof['plan_sha256'] != sha(plan_path) or sha(destination) != proof['data_sha256']
                        or sha(html_path) != proof['html_sha256']):
                    raise ValueError('Cached KIND detail changed')
                detail = json.loads(destination.read_text())
            else:
                cached = Path(probe_cache)/(event['event_id']+'.html') if probe_cache else None
                if cached is not None and cached.exists():
                    html = cached.read_text()
                else:
                    for attempt in range(3):
                        delay = interval-(time.monotonic()-last_request)
                        if delay > 0:
                            time.sleep(delay)
                        last_request = time.monotonic()
                        try:
                            response = session.post(URL, data=dict(method=event['detail_method'],
                                forward=event['detail_method'], bzProcsNo=event['event_id'], contnId=event['event_id']), timeout=30)
                            if response.status_code in [500, 502, 503, 504]:
                                raise requests.ConnectionError('Temporary KIND server error')
                            response.raise_for_status(); response.encoding = 'utf-8'; html = response.text
                            break
                        except (requests.ConnectionError, requests.Timeout):
                            if attempt == 2:
                                raise
                            time.sleep(2**attempt)
                detail = parse_detail(html, event)
                detail['retrieved_at'] = time.time()
                html_path.write_text(html); write_json(destination, detail)
                write_json(receipt, dict(plan_sha256=sha(plan_path), data_sha256=sha(destination), html_sha256=sha(html_path)))
            records.append(detail)
            if i % 20 == 0 or i+1 == len(events):
                progress = dict(at=time.time(), completed=i+1, requested=len(events),
                                elapsed_seconds=time.monotonic()-began, share_class_verified=False)
                write_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
    finally:
        session.close()
    write_json(output/'details.json', records)
    write_json(output/'complete.json', dict(at=time.time(), events=len(records),
        plan_sha256=sha(plan_path), details_sha256=sha(output/'details.json'), historical_listed_shares_certified=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', required=True); ap.add_argument('--cohort', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--probe-cache')
    args = ap.parse_args(); run(args.groups, args.cohort, args.output, probe_cache=args.probe_cache)
