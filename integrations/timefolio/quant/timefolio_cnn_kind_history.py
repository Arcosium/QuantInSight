"""Archive public KIND market-exit/issuance events in an isolated cache.

Issuer popup IDs are NOT converted into stock codes by appending a zero.
Current status icons are NOT historical trading-ban observations. Market exits
can be transfers between exchanges; they are not necessarily company failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import requests

from quant.timefolio_cnn_history import END, write_json, sha

BASE = 'https://kind.krx.co.kr'
ROUTES = dict(delisting='/investwarn/delcompany.do', issuance='/corpgeneral/stockissuelist.do')
METHODS = dict(delisting='searchDelCompanySub', issuance='searchStockIssueList')
FORWARDS = dict(delisting='delcompany_sub', issuance='searchStockIssueList')


def event_groups(records, kind):
    """Retain duplicate/conflicting occurrences instead of applying shares twice."""
    grouped = {}
    for row in records:
        key = (row['event_id'], row['issuer_popup_id']) if kind == 'issuance' else (
            row['issuer_popup_id'], row['date'], row['reason'])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, rows in grouped.items():
        clean = [{k:v for k,v in row.items() if k not in ['source_page','source_row']} for row in rows]
        conflicting = any(row != clean[0] for row in clean[1:])
        result.append(dict(identity=list(key), occurrences=len(rows), conflicting=conflicting,
            canonical_record=None if conflicting else clean[0],
            source_positions=[dict(page=r.get('source_page'), row=r.get('source_row')) for r in rows],
            ledger_application_certified=False))
    return result


def parse_page(html, kind, start, end):
    if kind not in ROUTES:
        raise ValueError('Unknown KIND history kind')
    soup = BeautifulSoup(html, 'html.parser')
    match = re.search(r'전체\s*([\d,]+)\s*건', soup.get_text(' ', strip=True))
    if not match:
        raise ValueError('Missing KIND pagination total; response is not a history page')
    total = int(match[1].replace(',', '')); records = []
    for row in soup.select('tbody tr'):
        cells = row.find_all('td', recursive=False)
        if len(cells) != (5 if kind == 'delisting' else 6):
            continue
        name_cell = cells[1 if kind == 'delisting' else 0]
        anchor = name_cell.find('a', onclick=re.compile('companysummary_open'))
        if anchor is None:
            raise ValueError('Missing issuer identifier')
        identifier = re.search(r"companysummary_open\('([^']+)'\)", anchor['onclick'])
        if not identifier:
            raise ValueError('Unrecognized issuer-popup identifier')
        day = cells[2 if kind == 'delisting' else 1].get_text(strip=True).replace('-', '')
        datetime.strptime(day, '%Y%m%d')
        if not start <= day <= end:
            raise ValueError('Event outside requested historical window')
        market = [i.get('alt') for i in name_cell.find_all('img')
                  if i.get('alt') in ['코스닥', '유가증권', '코넥스']]
        common = dict(issuer_popup_id=identifier[1], company_name=anchor.get_text(strip=True),
                      date=day, market_icons=market, security_code=None,
                      current_status_icons_used=False)
        if kind == 'delisting':
            reason = cells[3].get_text(' ', strip=True)
            common.update(reason=reason, event_type='market_exit',
                market_transfer_candidate=bool(re.search(r'(유가증권|코스닥|코넥스)\s*시장\s*상장', reason)),
                terminal_delisting_certified=False)
        else:
            event = re.search(r"fnDetailView(?:Etf)?\('([^']+)'\)", row.get('onclick', ''))
            if not event:
                raise ValueError('Missing issuance event identifier')
            quantity = cells[3].get_text(strip=True).replace(',', '')
            common.update(event_id=event[1], event_type='issuance_or_listing_change',
                listing_type=cells[2].get_text(strip=True), shares_raw=quantity,
                shares_signed=int(quantity) if re.fullmatch(r'-?\d+', quantity) else None,
                reason=cells[5].get_text(' ', strip=True),
                detail_method='searchEtfStockIssueDetail' if 'fnDetailViewEtf' in row.get('onclick', '') else 'searchStockIssueDetail',
                publication_time_verified=False, stock_class_verified=False)
        records.append(common)
    if total > 0 and not records:
        raise ValueError('Nonempty KIND history did not parse')
    return total, records


def collect(kind, start, end, output):
    if start > end or end > END:
        raise ValueError('Invalid/reserved date interval')
    for day in [start, end]:
        datetime.strptime(day, '%Y%m%d')
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(kind=kind, start=start, end=end, page_size=100, interval_seconds=1.,
                source=BASE+ROUTES[kind], module_sha256=sha(__file__))
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen KIND query changed')
    if not plan_path.exists():
        write_json(plan_path, plan)
    if (output/'complete.json').exists():
        old = json.loads((output/'complete.json').read_text())
        if any(sha(output/name) != digest for name, digest in old['artifacts'].items()):
            raise ValueError('Completed KIND archive changed')
        return old
    session = requests.Session(); records = []; page, expected_total = 1, None
    try:
        while True:
            path = output/f'page_{page:04d}.html'
            if path.exists():
                html = path.read_text()
            else:
                params = dict(method=METHODS[kind], forward=FORWARDS[kind], currentPageSize='100',
                    pageIndex=str(page), fromDate=datetime.strptime(start, '%Y%m%d').strftime('%Y-%m-%d'),
                    toDate=datetime.strptime(end, '%Y%m%d').strftime('%Y-%m-%d'),
                    marketType='' if kind == 'delisting' else 'all', tabType='1',
                    orderMode='1', orderStat='D', listingType='')
                r = session.post(BASE+ROUTES[kind], data=params, timeout=30)
                if r.status_code != 200:
                    raise RuntimeError(f'KIND HTTP {r.status_code}; no retry')
                r.encoding = 'utf-8'; html = r.text
                # Validate before saving unexpected error/login HTML as a data page.
                parse_page(html, kind, start, end)
                path.write_text(html)
                time.sleep(1.)
            total, batch = parse_page(html, kind, start, end)
            if expected_total is None:
                expected_total = total
            if total != expected_total or len(batch) != min(100, max(0, total-(page-1)*100)):
                raise ValueError('KIND page count/total changed during collection')
            records.extend({**r, 'source_page':page, 'source_row':i+1} for i,r in enumerate(batch))
            print(json.dumps(dict(kind=kind, page=page, pages=max(1, math.ceil(total/100)),
                                  records=len(records), total=total)), flush=True)
            if page >= max(1, math.ceil(total/100)):
                break
            page += 1
    finally:
        session.close()
    if len(records) != expected_total:
        raise ValueError('Missing KIND events across pages')
    grouped = event_groups(records, kind)
    write_json(output/'events.json', records)
    write_json(output/'event_groups.json', grouped)
    proof = dict(at=time.time(), kind=kind, start=start, end=end, total=len(records),
        distinct_event_identities=len(grouped), duplicate_groups=sum(g['occurrences']>1 for g in grouped),
        conflicting_groups=sum(g['conflicting'] for g in grouped),
        source_page_count=page, security_mapping_certified=False, point_in_time_universe_certified=False,
        source_files_written=False, artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'complete.json', proof)
    return proof


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=sorted(ROUTES))
    ap.add_argument('--start', default='20220101')
    ap.add_argument('--end', default=END)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); collect(args.kind, args.start, args.end, args.output)
