"""Public date-bounded daily reference, isolated from collectors and credentials.

Provider adjustment/venue conventions remain unverified. In particular, never
replace point-in-time turnover or market capitalisation with adjusted prices.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

import requests

from quant.timefolio_cnn_history import START, END, sha, write_json

URL = 'https://api.finance.naver.com/siseJson.naver'


def parse(text, start=START, end=END):
    value = ast.literal_eval(text.strip())
    if not isinstance(value, list) or not value or value[0][:6] != ['날짜','시가','고가','저가','종가','거래량']:
        raise ValueError('Unknown public daily-reference schema')
    result = []
    for row in value[1:]:
        if not isinstance(row, list) or len(row) != len(value[0]):
            raise ValueError('Malformed daily row')
        day = str(row[0])
        if not re.fullmatch(r'\d{8}', day) or not start <= day <= end:
            raise ValueError('Reference crossed the registered cutoff')
        datetime.strptime(day, '%Y%m%d')
        numbers = [float(v) for v in row[1:6]]
        if not all(math.isfinite(v) and v >= 0 for v in numbers):
            raise ValueError('Nonfinite or negative daily observation')
        result.append(dict(date=day, **dict(zip(['open','high','low','close','volume'],numbers))))
    dates = [r['date'] for r in result]
    if dates != sorted(set(dates)):
        raise ValueError('Duplicate or unsorted daily dates')
    return result


def collect(source_plan, output):
    source_plan, output = Path(source_plan), Path(output)
    codes = json.loads(source_plan.read_text())['codes']
    plan = dict(codes=codes, start=START, end=END, provider='NAVER', url=URL,
        interval_seconds=1., source_plan_sha256=sha(source_plan), module_sha256=sha(__file__),
        adjustment_basis_verified=False, venue_verified=False, historical_universe_verified=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'plan.json'
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError('Frozen acquisition plan changed')
    if not target.exists():
        write_json(target, plan)
    last, began = 0., time.monotonic()
    with requests.Session() as session:
        for index, code in enumerate(codes):
            if re.fullmatch(r'[0-9A-Z]{6}', code) is None:
                raise ValueError('Invalid security identifier')
            path, receipt = output/(code+'.json'), output/(code+'.receipt.json')
            if path.exists() or receipt.exists():
                if not path.exists() or not receipt.exists() or sha(path) != json.loads(receipt.read_text())['sha256']:
                    raise ValueError('Partial or changed reference artifact')
                continue
            for attempt in range(3):
                delay = 1-(time.monotonic()-last)
                if delay > 0:
                    time.sleep(delay)
                last=time.monotonic()
                try:
                    response=session.get(URL,params=dict(symbol=code,requestType='1',startTime=START,endTime=END,timeframe='day'),timeout=20)
                except requests.Timeout:
                    if attempt == 2:
                        raise RuntimeError('Public reference timed out after three attempts') from None
                    time.sleep(2**(attempt+1)); continue
                if response.status_code == 200:
                    break
                if response.status_code not in [500,502,503,504] or attempt == 2:
                    raise RuntimeError('Public reference HTTP '+str(response.status_code))
                time.sleep(2**(attempt+1))
            rows = parse(response.text)
            write_json(path,dict(provider='NAVER', code=code, start=START, end=END,
                retrieved_at=time.time(),response_sha256=hashlib.sha256(response.content).hexdigest(),
                adjustment_basis_verified=False,venue_verified=False, rows=rows))
            write_json(receipt,dict(code=code,rows=len(rows),sha256=sha(path),plan_sha256=sha(target)))
            progress=dict(at=time.time(),completed=index+1,requested=len(codes),last_code=code,
                last_rows=len(rows),seconds=time.monotonic()-began)
            write_json(output/'progress.json',progress)
            if index % 25 == 0:
                print(json.dumps(progress),flush=True)
    write_json(output/'complete.json',dict(at=time.time(),codes=codes,plan_sha256=sha(target)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-plan',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();collect(args.source_plan,args.output)
