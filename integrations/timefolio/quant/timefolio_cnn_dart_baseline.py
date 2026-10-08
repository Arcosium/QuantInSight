"""Read historical share baselines from already-authorized OpenDART access.

2021 annual reports are a possible opening baseline, not daily listed-share
counts. Filing availability and subsequent KIND listing changes still matter.
No authentication configuration or shared caches are changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import requests

from quant.timefolio_cnn_history import write_json, sha, END

URL = 'https://opendart.fss.or.kr/api/stockTotqySttus.json'


def normalize(rows):
    result = []
    for row in rows:
        number = str(row.get('istc_totqy', '')).replace(',', '').strip()
        receipt = str(row.get('rcept_no', ''))
        if not re.fullmatch(r'\d{14}', receipt):
            raise ValueError('Missing dated public-filing receipt')
        result.append(dict(share_class=row.get('se'), report_end=row.get('stlm_dt'),
            receipt=receipt, receipt_date=receipt[:8],
            issued_shares=int(number) if number.isdigit() else None,
            issued_shares_raw=row.get('istc_totqy'),
            available_before_cutoff=receipt[:8] <= END,
            listed_shares_certified=False))
    return result


def run(mapping, source_plan, output):
    key = os.environ.get('OPENDART_API_KEY') or os.environ.get('DART_API_KEY')
    if not key:
        raise RuntimeError('No existing OpenDART environment key; no credential edits attempted')
    rows = json.loads(mapping.read_text())['rows']
    wanted = set(json.loads(source_plan.read_text())['codes'])
    code_map = {}
    for row in rows:
        code = row['stock_code']
        if code in wanted:
            code_map.setdefault(code, set()).add(row['corp_code'])
    selected = {code:next(iter(corps)) for code, corps in code_map.items() if len(corps) == 1}
    if len(selected) > 3000:
        raise ValueError('Per-study baseline request budget exceeded')
    plan = dict(year='2021', report='11011', selected=selected,
        unmatched_codes=sorted(wanted-set(selected)), mapping_sha256=sha(mapping),
        source_plan_sha256=sha(source_plan), module_sha256=sha(__file__), interval_seconds=2.)
    output.mkdir(parents=True, exist_ok=True); plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen DART baseline plan changed')
    if not plan_path.exists():
        write_json(plan_path, plan)
    if (output/'complete.json').exists():
        old = json.loads((output/'complete.json').read_text())
        if any(sha(output/name) != digest for name, digest in old['artifacts'].items()):
            raise ValueError('Completed DART baseline changed')
        return old
    session = requests.Session(); previous = 0.; baseline_count = 0
    try:
        for index, (code, corp) in enumerate(sorted(selected.items())):
            dest = output/(code+'.json')
            if dest.exists():
                saved = json.loads(dest.read_text())
                if saved['code'] != code or saved['corp_code'] != corp:
                    raise ValueError('DART cached identity mismatch')
            else:
                delay = 2. - (time.monotonic()-previous)
                if delay > 0:
                    time.sleep(delay)
                previous = time.monotonic()
                response = session.get(URL, params=dict(crtfc_key=key, corp_code=corp,
                    bsns_year='2021', reprt_code='11011'), timeout=30)
                if response.status_code != 200:
                    raise RuntimeError('DART HTTP failure; no retry and request URL withheld')
                body = response.json(); status = body.get('status')
                if status not in ['000', '013']:
                    raise RuntimeError('DART refused baseline request; response withheld; no retry')
                saved = dict(code=code, corp_code=corp, year='2021', report='11011',
                    status=status, retrieved_at=time.time(), source=URL,
                    records=normalize(body.get('list', [])))
                write_json(dest, saved)
            baseline_count += any(row['issued_shares'] is not None
                and row['share_class'] == '보통주' and row['available_before_cutoff'] for row in saved['records'])
            progress = dict(at=time.time(), pid=os.getpid(), completed=index+1, requested=len(selected),
                common_share_baselines_found=baseline_count, daily_market_cap_ready=False)
            write_json(output/'progress.json', progress)
            if index % 25 == 0 or index+1 == len(selected):
                print(json.dumps(progress), flush=True)
    finally:
        session.close()
    write_json(output/'complete.json', dict(at=time.time(), codes=len(selected),
        common_share_baselines=baseline_count, daily_listed_shares_certified=False,
        artifacts={p.name:sha(p) for p in output.glob('*.json') if p.name not in ['progress.json','complete.json']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', type=Path, required=True)
    ap.add_argument('--source-plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); run(args.mapping, args.source_plan, args.output)
