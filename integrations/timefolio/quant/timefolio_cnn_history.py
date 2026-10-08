"""Bounded, read-only extraction of the Toss/KIS archive for a new CNN study.

Candidate regular-session aggregates are NOT certified exchange daily bars.
Venue, calendar, corporate-action and historical-universe audits remain explicit.
No imports from broker modules; no source database writes or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import closing
from datetime import datetime, timedelta
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time

VERSION = 'timefolio-cnn-history-v1'
DEFAULT_SOURCE = Path.home() / 'vault/CryptoBars/data/KRX'
DEFAULT_OUTPUT = Path.home() / 'vault/ArcTrade/timefolio_cnn_4y/20261002_v1'
START, CUTOVER, END = '20221123', '20250908', '20260923'
NXT_BOUNDARY = '20250304'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def connect(path):
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=3)
    c.execute('PRAGMA query_only=ON')
    c.execute('PRAGMA cache_size=-4096')
    c.execute('PRAGMA busy_timeout=3000')
    return c


def months(start=START, end=END):
    end_exclusive = (datetime.strptime(end, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
    if start > end:
        raise ValueError('Reversed date interval')
    year, month = int(start[:4]), int(start[4:6])
    while f'{year:04d}{month:02d}01' <= end:
        following = f'{year + (month == 12):04d}{month % 12 + 1:02d}01'
        yield max(start, f'{year:04d}{month:02d}01'), min(end_exclusive, following)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def canonical_hhmm(ts, source, auction_1531=False):
    if len(ts) != 14 or not ts.isdigit() or ts[12:] != '00':
        raise ValueError('Expected minute-aligned KST timestamp YYYYMMDDHHMMSS')
    hour, minute = int(ts[8:10]), int(ts[10:12])
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError('Invalid clock')
    if source == 'kis':
        return hour * 100 + minute
    if source != 'toss':
        raise ValueError('Unknown provider')
    # A non-NXT 15:30 label denotes the closing auction, not the 15:29 minute.
    if hour == 15 and minute == 30 and not auction_1531:
        return 1530
    elapsed = hour * 60 + minute - 1
    if elapsed < 0:
        raise ValueError('Toss bar crosses the date boundary')
    return elapsed // 60 * 100 + elapsed % 60


def aggregate_day(code, source, rows):
    if not rows:
        raise ValueError('Empty day')
    days = {str(r[0])[:8] for r in rows}
    if len(days) != 1:
        raise ValueError('A group must contain exactly one day')
    day = days.pop()
    if any(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1)):
        raise ValueError('Raw timestamps must be strictly ordered and unique')
    auction_1531 = source == 'toss' and any(r[0][8:] == '153100' for r in rows)
    regular, invalid, outside, duplicate = [], 0, 0, 0
    seen = set()
    for ts, o, h, l, c, v in rows:
        try:
            hhmm = canonical_hhmm(ts, source, auction_1531)
            if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in [o, h, l, c, v]):
                raise ValueError('Nonfinite bar')
            if not 0 < l <= min(o, c) <= max(o, c) <= h or v < 0:
                raise ValueError('Invalid OHLCV')
        except (ValueError, TypeError):
            invalid += 1
            continue
        if hhmm in seen:
            duplicate += 1
        seen.add(hhmm)
        if 900 <= hhmm <= 1530:
            regular.append((hhmm, o, h, l, c, v))
        else:
            outside += 1
    trading = [r for r in regular if r[5] > 0]
    execution = [r for r in trading if 905 <= r[0] <= 934]
    vol = sum(r[5] for r in trading)
    value = sum((r[2] + r[3] + r[4]) / 3 * r[5] for r in trading)
    ev = sum(r[5] for r in execution)
    evalue = sum((r[2] + r[3] + r[4]) / 3 * r[5] for r in execution)
    return dict(code=code, date=day, source=source,
        raw_first_ts=rows[0][0], raw_last_ts=rows[-1][0], raw_bars=len(rows),
        open=trading[0][1] if trading else None,
        high=max((r[2] for r in trading), default=None),
        low=min((r[3] for r in trading), default=None),
        close=trading[-1][4] if trading else None,
        volume=vol, typical_value_proxy=value,
        exec_price_proxy=evalue / ev if ev else None, exec_volume=ev,
        exec_bars=len(execution), regular_bars=len(regular), outside_bars=outside,
        invalid_bars=invalid, canonical_duplicates=duplicate,
        has_1530_trade=any(r[0] == 1530 for r in trading),
        late_open_candidate=bool(trading and trading[0][0] >= 1000),
        krx_venue_verified=source == 'kis' or day < NXT_BOUNDARY,
        session_calendar_verified=False, corporate_action_basis_verified=False,
        historical_universe_certified=False)


def read_month(path, code, lo, hi):
    # Close the snapshot before aggregation; do not pin a collector's WAL.
    with closing(connect(path)) as c:
        rows = c.execute('SELECT ts,o,h,l,c,v FROM bars WHERE code=? AND ts>=? AND ts<? ORDER BY ts',
                         (code, lo + '000000', hi + '000000')).fetchall()
    return rows


def grouped_days(rows):
    group, day = [], None
    for row in rows:
        next_day = row[0][:8]
        if day is not None and next_day != day:
            yield group
            group = []
        group.append(row); day = next_day
    if group:
        yield group


def preflight(source, output):
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    all_codes = set()
    for name in ['bars_toss.db', 'bars_ohlc.db']:
        path = source / name
        with closing(connect(path)) as c:
            schema = c.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            codes = [r[0] for r in c.execute('SELECT DISTINCT code FROM done ORDER BY code')]
            all_codes.update(codes)
            plan = c.execute('EXPLAIN QUERY PLAN SELECT ts,o,h,l,c,v FROM bars WHERE code=? AND ts>=? AND ts<? ORDER BY ts',
                             ('005930', START + '000000', END + '999999')).fetchall()
            assert any('SEARCH bars USING INDEX' in r[-1] for r in plan)
            meta = dict(path=str(path), bytes=path.stat().st_size, schema=schema, codes=len(codes),
                        journal_mode=c.execute('PRAGMA journal_mode').fetchone()[0], query_plan=plan)
            if name == 'bars_toss.db':
                done = c.execute('SELECT code,status FROM done ORDER BY code').fetchall()
                meta.update(done_ledger_sha256=hashlib.sha256(json.dumps(done).encode()).hexdigest(),
                    declared_bars=sum(int(v) for _, v in done if v != 'notfound'),
                    notfound_codes=[code for code, v in done if v == 'notfound'],
                    zero_row_codes=[code for code, v in done if v == '0'])
            sources[name] = meta
    plan = dict(version=VERSION, start=START, cutoff=END, cutover=CUTOVER,
        source_directory=str(source.resolve()), codes=sorted(all_codes),
        extractor_sha256=sha(__file__), sources=sources,
        interpretation='Candidate regular-session aggregates; not training-ready or contest-certified.',
        missing_gates=['Dated session calendar and special opens/closes',
            'KRX-only daily reference for provider/venue and corporate-action checks',
            'Historical eligibility, delistings and GICS sector weights',
            'Current contest rule verification before account simulation'],
        reserved_after_cutoff_unread=True, original_databases_modified=False)
    target = output / 'source_plan.json'
    if target.exists():
        previous = json.loads(target.read_text())
        for key in ['version', 'start', 'cutoff', 'cutover', 'source_directory', 'codes', 'extractor_sha256']:
            if previous[key] != plan[key]:
                raise ValueError('Frozen extraction plan changed; choose a new output directory')
        return previous
    write_json(target, plan)
    return plan


def extract(source, output, codes):
    plan = preflight(source, output)
    if any(code not in plan['codes'] for code in codes):
        raise ValueError('Unknown code')
    shards = output / 'daily_candidates'; shards.mkdir(exist_ok=True)
    began, total_rows = time.monotonic(), 0
    for index, code in enumerate(codes):
        target, receipt_path = shards / f'{code}.csv.gz', shards / f'{code}.json'
        if target.exists() or receipt_path.exists():
            if not target.exists() or not receipt_path.exists():
                raise ValueError(f'Partial existing artifact for {code}; review before resuming')
            proof = json.loads(receipt_path.read_text())
            if proof['plan_sha256'] != sha(output / 'source_plan.json') or proof['sha256'] != sha(target):
                raise ValueError(f'Existing artifact mismatch: {code}')
            continue
        daily, raw_rows, source_digests = [], 0, {}
        for provider, filename, start, end in [('toss', 'bars_toss.db', START, '20250905'),
                                               ('kis', 'bars_ohlc.db', CUTOVER, END)]:
            stream_hash = hashlib.sha256()
            for lo, hi in months(start, end):
                rows = read_month(source / filename, code, lo, hi)
                raw_rows += len(rows)
                for group in grouped_days(rows):
                    daily.append(aggregate_day(code, provider, group))
                # Hash all input values, including rejected and extended-session bars.
                stream_hash.update(json.dumps(rows, separators=(',', ':'), allow_nan=False).encode())
            source_digests[provider] = stream_hash.hexdigest()
        if len({row['date'] for row in daily}) != len(daily):
            raise ValueError('Overlapping provider dates')
        temp = target.with_suffix(target.suffix + '.tmp')
        with temp.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0, filename='') as gz:
                with io.TextIOWrapper(gz, encoding='utf-8', newline='') as text:
                    if daily:
                        writer = csv.DictWriter(text, fieldnames=list(daily[0]))
                        writer.writeheader(); writer.writerows(daily)
        temp.replace(target)
        proof = dict(code=code, days=len(daily), raw_rows=raw_rows, sha256=sha(target),
            plan_sha256=sha(output / 'source_plan.json'), raw_input_hashes=source_digests,
            first_day=daily[0]['date'] if daily else None, last_day=daily[-1]['date'] if daily else None,
            unverified_venue_days=sum(not r['krx_venue_verified'] for r in daily),
            invalid_bars=sum(r['invalid_bars'] for r in daily),
            duplicate_canonical_minutes=sum(r['canonical_duplicates'] for r in daily),
            missing_auction_days=sum(not r['has_1530_trade'] for r in daily),
            late_open_candidate_days=sum(r['late_open_candidate'] for r in daily),
            training_ready=False, contest_certified=False)
        write_json(receipt_path, proof); total_rows += raw_rows
        progress = dict(at=time.time(), pid=os.getpid(), completed=index + 1, requested=len(codes),
            last_code=code, raw_rows_processed_this_process=total_rows,
            elapsed_seconds=time.monotonic() - began, training_started=False)
        write_json(output / 'extraction_progress.json', progress)
        print(json.dumps(progress), flush=True)
    write_json(output / 'extraction_complete.json', dict(at=time.time(), codes=codes,
        plan_sha256=sha(output / 'source_plan.json'), training_ready=False,
        seconds=time.monotonic() - began, artifacts={code: sha(shards / f'{code}.json') for code in codes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['preflight', 'extract'])
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--codes', nargs='+')
    args = ap.parse_args()
    plan = preflight(args.source, args.output)
    if args.command == 'extract':
        extract(args.source, args.output, args.codes or plan['codes'])
    else:
        print(json.dumps(dict(codes=len(plan['codes']), output=str(args.output), training_ready=False)))


if __name__ == '__main__':
    main()
