"""Separate KIS historical-reference cache with read-only shared credentials.

GET quotations only. Never imports the live broker, refreshes a token, edits its
cache, writes the shared database, or changes any collector. output1 contains
current snapshots and is discarded; it cannot supply historical market caps.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import requests
from cryptography.fernet import Fernet

from quant.timefolio_cnn_history import START, END, write_json, sha

BASE = 'https://openapi.koreainvestment.com:9443'
API = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
TR = 'FHKST03010100'


class ReferenceTemporarilyUnavailable(RuntimeError):
    """Bounded transport/server failure; another code can still be collected."""


def existing_credentials(private):
    """Match the collector's admin-owned real profile without initializing it."""
    with closing(sqlite3.connect((private/'arquant_auth.db').resolve().as_uri()+'?mode=ro', uri=True)) as c:
        c.execute('PRAGMA query_only=ON')
        rows = c.execute('SELECT id,kis_app_key_enc,kis_app_secret_enc,kis_base_url FROM users '
            'WHERE is_admin=1 OR owner_id IN (SELECT id FROM users WHERE is_admin=1) ORDER BY id').fetchall()
    cipher = Fernet((private/'.fernet.key').read_bytes())
    for uid, encrypted_key, encrypted_secret, base in rows:
        if base and base.rstrip('/') != BASE:
            continue
        token_path = private/str(uid)/'kis_token.json'
        if not token_path.exists() or not encrypted_key or not encrypted_secret:
            continue
        cache = json.loads(token_path.read_text())
        key = cipher.decrypt(encrypted_key.encode()).decode()
        if cache.get('appkey') != key or float(cache.get('expires_at', 0)) <= time.time()+600:
            continue
        return dict(key=key, secret=cipher.decrypt(encrypted_secret.encode()).decode(),
                    token=cache['access_token'], token_path=token_path)
    raise RuntimeError('No valid existing KIS real token; no token refresh attempted')


def page_rows(response, start, cursor):
    if response.get('rt_cd') != '0':
        # Response bodies and provider messages can contain account context.
        raise RuntimeError('KIS quotation refused; response withheld; no automatic retry')
    bars = response.get('output2')
    if not isinstance(bars, list):
        raise ValueError('Missing historical daily records')
    dates = []
    for bar in bars:
        day = bar.get('stck_bsop_date', '')
        if len(day) != 8 or not day.isdigit() or not start <= day <= cursor:
            raise ValueError('Historical response crossed the requested date bounds')
        datetime.strptime(day, '%Y%m%d')
        for name in ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol', 'acml_tr_pbmn']:
            if name not in bar:
                raise ValueError('Missing OHLCV/actual-value reference field')
        dates.append(day)
    if len(set(dates)) != len(dates):
        raise ValueError('Duplicate reference dates')
    return bars


def next_cursor(bars, start):
    if not bars:
        return None
    first = min(b['stck_bsop_date'] for b in bars)
    if first <= start:
        return None
    return (datetime.strptime(first, '%Y%m%d')-timedelta(days=1)).strftime('%Y%m%d')


class ReadOnlyKIS:
    def __init__(self, private, interval=2.):
        if interval < 2:
            raise ValueError('This study is limited to at most one KIS request per two seconds')
        self.private, self.interval, self.last = private, interval, 0.
        self.session = requests.Session()

    def fetch(self, code, start, cursor):
        # Re-read the token to follow a legitimate collector refresh, never issue one.
        credentials = existing_credentials(self.private)
        delay = self.interval - (time.monotonic()-self.last)
        if delay > 0:
            time.sleep(delay)
        headers = dict(authorization='Bearer '+credentials['token'], appkey=credentials['key'],
            appsecret=credentials['secret'], tr_id=TR, custtype='P')
        params = dict(FID_COND_MRKT_DIV_CODE='J', FID_INPUT_ISCD=code,
            FID_INPUT_DATE_1=start, FID_INPUT_DATE_2=cursor, FID_PERIOD_DIV_CODE='D', FID_ORG_ADJ_PRC='1')
        for attempt in range(3):
            self.last = time.monotonic()
            try:
                r = self.session.get(BASE+API, headers=headers, params=params, timeout=30)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise ReferenceTemporarilyUnavailable('KIS transport failed after three bounded attempts') from None
                time.sleep(2 ** (attempt+1)); continue
            if r.status_code == 200:
                return page_rows(r.json(), start, cursor)
            if r.status_code not in [500, 502, 503, 504]:
                raise RuntimeError(f'KIS HTTP {r.status_code}; request stopped')
            if attempt == 2:
                raise ReferenceTemporarilyUnavailable(f'KIS HTTP {r.status_code} after three attempts')
            time.sleep(2 ** (attempt+1))


def collect_code(client, output, code, start=START, end=END):
    if re.fullmatch(r'[0-9A-Z]{6}', code) is None or start > end or end > END:
        raise ValueError('Registered stock code and non-reserved interval required')
    root = output/code; root.mkdir(parents=True, exist_ok=True)
    receipt = root/'complete.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['code'] != code or old['start'] != start or old['end'] != end:
            raise ValueError('Frozen reference interval changed')
        if any(sha(root/name) != digest for name, digest in old['pages'].items()):
            raise ValueError('Reference-page fingerprint changed')
        return old
    cursor, pages, rows, days = end, {}, 0, set()
    while cursor:
        path = root/f'{cursor}.json'
        if path.exists():
            page = json.loads(path.read_text())
            if (page['code'], page['start'], page['cursor'], page['venue'], page['unadjusted']) != (code, start, cursor, 'J', True):
                raise ValueError('Cached page belongs to a different query')
            bars = page_rows({'rt_cd':'0', 'output2':page['bars']}, start, cursor)
        else:
            bars = client.fetch(code, start, cursor)
            page = dict(provider='KIS', code=code, start=start, cursor=cursor, venue='J', unadjusted=True,
                retrieved_at=time.time(), bars=bars, discarded_current_output1=True)
            write_json(path, page)
        page_days = {bar['stck_bsop_date'] for bar in bars}
        if days & page_days:
            raise ValueError('Overlapping historical pages')
        days |= page_days; rows += len(bars); pages[path.name] = sha(path)
        cursor = next_cursor(bars, start)
    result = dict(code=code, start=start, end=end, rows=rows, pages=pages,
        first=min(days) if days else None, last=max(days) if days else None,
        complete_requests=True, all_sessions_certified=False, corporate_actions_reconciled=False,
        historical_market_cap_available=False, gics_available=False,
        shared_token_refreshed=False, shared_files_written=False)
    write_json(receipt, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--codes', nargs='+')
    ap.add_argument('--source-plan', type=Path)
    args = ap.parse_args()
    if bool(args.codes) == bool(args.source_plan):
        raise ValueError('Supply codes or one frozen source plan')
    codes = args.codes or json.loads(args.source_plan.read_text())['codes']
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/'plan.json'
    plan = dict(codes=codes, start=START, end=END, interval_seconds=2., module_sha256=sha(__file__),
        source_plan_sha256=sha(args.source_plan) if args.source_plan else None)
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Reference acquisition plan changed; choose a new output directory')
    if not plan_path.exists():
        write_json(plan_path, plan)
    client = ReadOnlyKIS(Path.home()/'vault/QuantInSight/data')
    began = time.monotonic(); failed=[]; succeeded=0
    try:
        for index, code in enumerate(codes):
            try:
                result = collect_code(client, args.output, code)
            except ReferenceTemporarilyUnavailable as error:
                failed.append(dict(code=code,at=time.time(),reason=str(error)))
                write_json(args.output/'deferred_codes.json',failed)
                continue
            succeeded+=1
            progress = dict(at=time.time(), pid=os.getpid(), completed=index+1, requested=len(codes),
                successful_codes=succeeded, deferred_codes=len(failed),
                last_code=code, last_rows=result['rows'], elapsed_seconds=time.monotonic()-began,
                token_refresh_attempted=False, training_started=False)
            write_json(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    finally:
        client.session.close()
    write_json(args.output/'deferred_codes.json',failed)
    filename='pass_complete_with_deferred.json' if failed else 'complete.json'
    write_json(args.output/filename, dict(at=time.time(), codes=codes,successful_codes=succeeded,
        deferred=failed,all_codes_complete=not failed,plan_sha256=sha(plan_path)))


if __name__ == '__main__':
    main()
