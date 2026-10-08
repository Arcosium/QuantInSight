"""Assess completed boundary weeks using verified exchange closure dates.

The frozen ledger excluded both boundary weeks regardless of the calendar.
Interior holiday weeks keep their original treatment. Known closures resolve
whether a boundary week extends beyond the observed evaluation interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_stock_limits import PRIOR_NAMES
from quant.timefolio_heatmap_walkforward import SOURCE

DEST = SOURCE.with_name('20260929_week_boundary_audit_v1')
CLOSED = frozenset(['20260924', '20260925'])
OFFICIAL_URL = 'https://kind.krx.co.kr/external/dst/notice/11637/%5B%ED%95%9C%EA%B5%AD%EA%B1%B0%EB%9E%98%EC%86%8C%5D%202026%EB%85%84%20%EC%98%AC%EB%B9%BC%EB%AF%B8%EA%B3%B5%EC%8B%9C%20%EC%95%88%EB%82%B4.pdf'


def assess_weeks(result, known_closed=CLOSED):
    dates = [row['date'] for row in result['daily']]
    if not dates or dates != sorted(set(dates)): raise ValueError('Ordered daily observations required')
    first, last = dates[0], dates[-1]; assessed = []
    for row in result['weekly']:
        period = pd.Period(row['week'], freq='W-SUN')
        expected = [d.strftime('%Y%m%d') for d in pd.bdate_range(period.start_time, period.end_time)
                    if d.strftime('%Y%m%d') not in known_closed]
        # Only boundary coverage changes; missing interior weekdays may be other
        # exchange holidays, whose weeks were already included in the ledger.
        complete = bool(expected) and expected[0] >= first and expected[-1] <= last
        if complete: assessed.append(row)
    low = sum(row['turnover'] < .05 for row in assessed)
    old = result['metrics']
    return {'interior_only_low_turnover_weeks': old.get('interior_only_low_turnover_weeks', old['low_turnover_weeks']),
            'interior_only_assessed_full_weeks': old.get('interior_only_assessed_full_weeks', old['assessed_full_weeks']),
            'low_turnover_weeks': int(low), 'assessed_full_weeks': len(assessed),
            'four_week_turnover_stop': bool(low >= 4),
            'calendar_assessed_weeks': [row['week'] for row in assessed],
            'boundary_calendar_source': OFFICIAL_URL}


def run():
    DEST.mkdir(exist_ok=True)
    spec = {'source_url': OFFICIAL_URL, 'known_closed_weekdays': sorted(CLOSED), 'roots': PRIOR_NAMES,
            'scope': '2026-09-21..23 is the full last trading week; include it. Preserve partial first week exclusion and interior holiday-week treatment. Returns and orders unchanged.',
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Week audit protocol changed')
    if not path.exists(): atomic_json(path, spec)
    rows = []
    for name in PRIOR_NAMES:
        for path in sorted((SOURCE.with_name(name)/'portfolios').glob('*.json')):
            result = json.loads(path.read_text()); fixed = assess_weeks(result)
            rows.append(dict(root=name, id=path.stem,
                             previous_low=result['metrics']['low_turnover_weeks'],
                             corrected_low=fixed['low_turnover_weeks'],
                             previous_stop=result['metrics']['four_week_turnover_stop'],
                             corrected_stop=fixed['four_week_turnover_stop'],
                             assessed_weeks=fixed['assessed_full_weeks'],
                             last_week=result['weekly'][-1]['week'], last_week_turnover=result['weekly'][-1]['turnover']))
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'corrected_turnover.csv', index=False)
    summary = {'portfolios': len(rows), 'additional_low_week': int((frame.corrected_low > frame.previous_low).sum()),
               'new_turnover_failures': int((frame.corrected_stop & ~frame.previous_stop).sum()),
               'previously_failed_becoming_pass': int((~frame.corrected_stop & frame.previous_stop).sum()),
               'source_url': OFFICIAL_URL, 'returns_and_trades_unchanged': True}
    atomic_json(DEST/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['audit-prior']); ap.parse_args(); run()
