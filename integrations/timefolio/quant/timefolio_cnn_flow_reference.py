"""Isolated net-share snapshots; shared collectors and reserved prices stay closed."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import time

import requests

START, END = '20221123', '20260923'
URL = 'https://m.stock.naver.com/api/stock/{code}/trend'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def bounded_day(value):
    day = str(value).replace('-', '')
    if not re.fullmatch(r'\d{8}', day):
        raise ValueError('Invalid flow date')
    datetime.strptime(day, '%Y%m%d')
    return day


def quantity(value):
    value = str(value).replace(',', '')
    if not re.fullmatch(r'[+-]?\d+', value):
        raise ValueError('Missing or non-integer net-share amount')
    return int(value)


def parse(rows, *, mobile=False, code=None):
    result = {}
    future = before = duplicates = 0
    for row in rows:
        day = bounded_day(row.get('bizdate' if mobile else 'date'))
        # Filter before touching any numerical value, including flow values.
        if day > END:
            future += 1
            continue
        if day < START:
            before += 1
            continue
        if mobile and row.get('itemCode', code) != code:
            raise ValueError('Flow response belongs to another security')
        value = dict(date=day,
            institution_net_shares=quantity(row.get('organPureBuyQuant' if mobile else 'inst_net')),
            foreign_net_shares=quantity(row.get('foreignerPureBuyQuant' if mobile else 'foreign_net')))
        if day in result:
            if value != result[day]:
                raise ValueError('Conflicting duplicate flow date')
            duplicates += 1
        result[day] = value
    return dict(rows=[result[day] for day in sorted(result)], discarded_future_rows=future,
        discarded_before_start_rows=before, identical_duplicate_rows=duplicates)


def archive(path):
    path = Path(path)
    for attempt in range(3):
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            break
        time.sleep(.5)
    else:
        raise RuntimeError('Shared flow file changed during all bounded read attempts')
    reader = csv.DictReader(StringIO(raw.decode('utf-8-sig')))
    if not {'date', 'inst_net', 'foreign_net'} <= set(reader.fieldnames or []):
        raise ValueError('Missing required archive fields')
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(), source_mtime_ns=after.st_mtime_ns,
        source_bytes=len(raw), **parse(reader))


def merge(archived, incoming):
    old = {row['date']: row for row in archived}
    new = {row['date']: row for row in incoming}
    overlap = old.keys() & new.keys()
    changed = sorted(day for day in overlap if old[day] != new[day])
    # A unit/basis mismatch must be reviewed before either series is substituted.
    combined = old if changed else {**old, **new}
    return dict(rows=[combined[day] for day in sorted(combined)], overlap_rows=len(overlap),
        conflicting_overlap_dates=changed, fresh_rows_accepted=not changed,
        new_dates_added=len(new.keys()-old.keys()) if not changed else 0)


def collect(audit_path, output):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    audit = json.loads(audit_path.read_text())
    source = Path(audit['source_directory']).resolve()
    if output.is_relative_to(source):
        raise ValueError('Research output must be outside the shared source directory')
    codes = sorted(r['code'] for r in audit['records'] if r['status'] == 'inspected')
    if not codes or len(codes) != len(set(codes)) or any(not re.fullmatch(r'[0-9A-Z]{6}', c) for c in codes):
        raise ValueError('Frozen, unique stock codes required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'archive').mkdir()
    (output/'merged').mkdir()
    plan = dict(start=START, end=END, codes=codes, archive_audit_sha256=sha(audit_path),
        source_directory=str(source), source_sha256=sha(__file__), url=URL,
        page_size=60, interval_seconds=2., fresh_request_limit=len(codes),
        authenticated_requests=0, shared_writes=False, price_fields_retained=False,
        missing_flow_policy='Absent observations remain absent; no filling or extrapolation.',
        historical_vintages_and_publication_time_verified=False,
        fund_or_national_pension_series_available=False)
    write(output/'plan.json', plan)
    snapshots = {}
    for code in codes:
        snapshot = archive(source/f'investor_{code}.csv')
        write(output/'archive'/f'{code}.json', dict(code=code, **snapshot))
        snapshots[code] = sha(output/'archive'/f'{code}.json')
    write(output/'archive_receipt.json', dict(at=time.time(), artifacts=snapshots,
        source_files_copied_with_date_and_field_filter=True, source_files_modified=False))
    last, began = 0., time.monotonic()
    summaries = []
    rate_limited = False
    with requests.Session() as session:
        for code in codes:
            old = json.loads((output/'archive'/f'{code}.json').read_text())
            fresh = dict(rows=[])
            failure = 'stopped_after_http429' if rate_limited else None
            response_sha = None
            if not rate_limited:
                delay = 2. - (time.monotonic()-last)
                if delay > 0:
                    time.sleep(delay)
                last = time.monotonic()
                try:
                    response = session.get(URL.format(code=code), params=dict(pageSize=60),
                        headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
                    if response.status_code != 200:
                        failure = f'http_{response.status_code}'
                        rate_limited = response.status_code == 429
                    else:
                        response_sha = hashlib.sha256(response.content).hexdigest()
                        value = response.json()
                        if not isinstance(value, list):
                            raise ValueError('Historical flow list required')
                        fresh = parse(value, mobile=True, code=code)
                except requests.RequestException:
                    failure = 'bounded_transport_failure'
                except (ValueError, TypeError, KeyError):
                    failure = 'response_schema_or_identity_or_duplicate_failure'
            combined = merge(old['rows'], fresh['rows'])
            summary = dict(code=code, fresh_failure=failure,
                fresh_response_sha256=response_sha,
                archive_sha256=snapshots[code],
                fresh_discarded_future_rows=fresh.get('discarded_future_rows', 0),
                fresh_rows=len(fresh['rows']), **{k:v for k,v in combined.items() if k != 'rows'})
            if failure:
                summary['fresh_rows_accepted'] = False
            record = dict(**summary, rows=combined['rows'])
            write(output/'merged'/f'{code}.json', record)
            summary.update(merged_sha256=sha(output/'merged'/f'{code}.json'),
                observations=len(combined['rows']),
                first=combined['rows'][0]['date'] if combined['rows'] else None,
                last=combined['rows'][-1]['date'] if combined['rows'] else None)
            summaries.append(summary)
            progress = dict(at=time.time(), state='collecting_recent_flow', completed=len(summaries),
                planned=len(codes), request_failures=sum(x['fresh_failure'] is not None for x in summaries),
                overlap_conflicts=sum(bool(x['conflicting_overlap_dates']) for x in summaries),
                seconds=time.monotonic()-began)
            write(output/'progress.json', progress)
            if len(summaries) % 25 == 0:
                print(json.dumps(progress), flush=True)
    write(output/'complete.json', dict(at=time.time(), plan_sha256=sha(output/'plan.json'),
        archive_receipt_sha256=sha(output/'archive_receipt.json'), summaries=summaries,
        research_pass_complete=True, entire_four_year_market_coverage_verified=False))
    write(output/'progress.json', dict(**{k:v for k,v in progress.items() if k != 'state'},
        state='research_pass_complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(args.audit, args.output)
