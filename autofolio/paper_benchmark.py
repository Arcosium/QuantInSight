"""Read-only paper reference; never admitted to the 36-month research population."""
import json
import math
import statistics
from datetime import date
from functools import lru_cache
from pathlib import Path

from .catalogue import strategy_case
from .metrics import ledger

SOURCE = Path.home() / 'projects/HYFE_QTPA/work/results'


def paper_returns(root=SOURCE):
    folds = []
    for index in range(4):
        path = root / f'ens_heatf10_direct_4h_s{index}_cohort_H84_wev.json'
        daily = sorted(json.loads(path.read_text())['daily'].items())
        if len(daily) < 2:
            raise ValueError('Empty paper fold')
        for day, nav in daily:
            date.fromisoformat(day)
            if not isinstance(nav, (float, int)) or not math.isfinite(nav) or nav <= 0:
                raise ValueError('Invalid paper NAV')
        returns = {day.replace('-', ''): nav / daily[i-1][1] - 1
                   for i, (day, nav) in enumerate(daily) if i}
        folds.append((daily[1][0], daily[-1][0], returns))
    # Latest-starting validation fold wins on overlap, identically for all comparisons.
    combined = {}
    for _, _, values in sorted(folds):
        combined.update(values)
    return dict(sorted(combined.items())), [dict(start=a, end=b) for a, b, _ in sorted(folds)]


def measure(returns):
    monthly = {}
    for day, value in sorted(returns.items()):
        if not math.isfinite(value) or value <= -1:
            raise ValueError('Invalid return')
        monthly[day[:6]] = monthly.get(day[:6], 1.) * (1 + value)
    return dict(net_return=math.prod(1 + r for r in returns.values()) - 1,
                negative_months=sum(v < 1 for v in monthly.values()), months=len(monthly))


@lru_cache(maxsize=2048)
def phase_measure(identity, digest, phase, days):
    # Source digest invalidates cached metrics when the catalogue refreshes a result.
    _, case = strategy_case(identity, phase)
    daily = {r['date']: r['net_daily_return'] for r in ledger(case)}
    return measure({day: daily[day] for day in days})


def comparison(rows):
    try:
        returns, folds = paper_returns()
        reference = dict(id='paper-heatf-event', title='논문 · HeatF CNN 가중', benchmark=True,
                         **measure(returns))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(available=False, message='논문 원본 기록을 확인할 수 없습니다.')
    points, skipped = [reference], 0
    for row in rows:
        try:
            phases = []
            for phase in range(row['phase_count']):
                phases.append(phase_measure(row['id'], row.get('source_digest'), phase, tuple(returns)))
            if not phases:
                raise ValueError('Missing phases')
            points.append(dict(id=row['id'], title=row['title'], benchmark=False,
                               **{key: statistics.mean(p[key] for p in phases)
                                  for key in ('net_return', 'negative_months', 'months')}))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            skipped += 1
    return dict(available=True, points=points, folds=folds, days=len(returns), skipped=skipped,
                start=min(returns), end=max(returns),
                superior=sum(p['net_return'] > reference['net_return'] and
                             p['negative_months'] <= reference['negative_months'] or
                             p['net_return'] >= reference['net_return'] and
                             p['negative_months'] < reference['negative_months'] for p in points[1:]))
