"""Resolve public KIND issuer IDs without inventing ticker check digits.

Only security identifiers and listing metadata are saved. Current company
status or management details are not a point-in-time eligibility history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import requests

from quant.timefolio_cnn_history import sha, write_json

URL = 'https://kind.krx.co.kr/common/companysummary.do'


def parse_issuer(html, issuer_id):
    soup = BeautifulSoup(html, 'html.parser')
    allowed = {'표준코드':'isin', '종목코드':'security_code', '상장일':'listing_date',
               '상장일자':'listing_date', '시장구분':'market', '회사명':'company_name'}
    out = dict(issuer_popup_id=issuer_id)
    for heading in soup.find_all('th'):
        name = heading.get_text(' ', strip=True).replace(' ', '')
        cell = heading.find_next_sibling('td')
        if name in allowed and cell is not None:
            value = cell.get_text(' ', strip=True)
            field = allowed[name]
            if field in out and out[field] != value:
                raise ValueError('Conflicting security identifiers')
            out[field] = value
    if not re.fullmatch(r'[0-9A-Z]{6}', out.get('security_code', '')):
        raise ValueError('No unambiguous six-character security code')
    if not re.fullmatch(r'[A-Z]{2}[0-9A-Z]{10}', out.get('isin', '')):
        raise ValueError('Missing ISIN')
    out.update(current_snapshot=True, historical_identity_certified=False,
               source=URL, retrieved_at=time.time())
    return out


def run(events_path, output, existing_plan):
    events = json.loads(events_path.read_text())
    issuer_ids = sorted({r['issuer_popup_id'] for r in events})
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(events_sha256=sha(events_path), existing_plan_sha256=sha(existing_plan),
        issuers=issuer_ids, source_sha256=sha(__file__))
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Issuer resolution plan changed')
    if not plan_path.exists():
        write_json(plan_path, plan)
    records = []; missing = []; session = requests.Session()
    try:
        for index, issuer in enumerate(issuer_ids):
            dest = output/(issuer+'.json')
            if dest.exists():
                result = json.loads(dest.read_text())
                if result.get('issuer_popup_id') != issuer:
                    raise ValueError('Cached issuer mismatch')
            else:
                r = session.post(URL, data=dict(method='searchCompanySummaryOvrvwDetail',
                    strIsurCd=issuer, menuIndex='0', lstCd='undefined', methodType='0'), timeout=30)
                if r.status_code != 200:
                    raise RuntimeError(f'KIND issuer HTTP {r.status_code}; no retry')
                r.encoding = 'utf-8'
                try:
                    result = parse_issuer(r.text, issuer)
                except ValueError as e:
                    # Missing issuer identity is retained for a coverage audit.
                    result = dict(issuer_popup_id=issuer, unresolved_reason=str(e),
                                  retrieved_at=time.time(), source=URL)
                result['public_html_sha256'] = __import__('hashlib').sha256(r.content).hexdigest()
                write_json(dest, result); time.sleep(1.)
            if 'security_code' in result:
                records.append(result)
            else:
                missing.append(issuer)
            write_json(output/'progress.json', dict(at=time.time(), completed=index+1,
                requested=len(issuer_ids), resolved=len(records), unresolved=len(missing)))
            if index % 20 == 0 or index+1 == len(issuer_ids):
                print(json.dumps(dict(completed=index+1, requested=len(issuer_ids),
                    resolved=len(records), unresolved=len(missing))), flush=True)
    finally:
        session.close()
    existing = set(json.loads(existing_plan.read_text())['codes'])
    resolved = {r['issuer_popup_id']:r for r in records}
    absent = [{**event, 'security_code':resolved[event['issuer_popup_id']]['security_code'],
               'isin':resolved[event['issuer_popup_id']]['isin']}
              for event in events if event['issuer_popup_id'] in resolved
              and resolved[event['issuer_popup_id']]['security_code'] not in existing]
    write_json(output/'absent_from_backfill_events.json', absent)
    write_json(output/'complete.json', dict(at=time.time(), resolved=len(records), unresolved=missing,
        absent_codes=sorted({r['security_code'] for r in absent}),
        history_universe_certified=False, plan_sha256=sha(plan_path),
        artifacts={p.name:sha(p) for p in output.glob('*.json') if p.name not in ['progress.json','complete.json']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--existing-plan', type=Path, required=True)
    args = ap.parse_args(); run(args.events, args.output, args.existing_plan)
