from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from quant.timefolio_heatmap_morning_news import morning_signals
from quant.timefolio_heatmap_macro_overlay import schedules_from_records
from quant.timefolio_heatmap_macro_archive import KST
from test_timefolio_macro_archive import record


def test_morning_deadline_is_next_session_and_excludes_later_report():
    records = [record('2026-05-18T16:00:00+09:00', 'close', .5),
               record('2026-05-19T08:55:00+09:00', 'morning', .4),
               record('2026-05-19T08:55:01+09:00', 'late', .8)]
    out = morning_signals(records, ['20260518', '20260519', '20260520'])
    assert out[0]['execution_date'] == '20260519' and out[0]['report_sha256'] == 'morning'
    assert out[0]['age_hours'] == 0 and out[1]['report_sha256'] == 'late'
    assert out[-1]['execution_date'] is None and not out[-1]['available']


def test_weekend_clock_age_and_cached_repetitions_are_respected():
    records = [record('2026-05-22T08:55:00+09:00', 'cached', .4),
               record('2026-05-25T08:00:00+09:00', 'cached', .4)]
    out = morning_signals(records, ['20260522', '20260525', '20260526', '20260527', '20260528'])
    assert out[0]['age_hours'] == 72 and out[1]['age_hours'] == 96
    assert not out[2]['available']


def test_future_reports_and_added_calendar_tail_preserve_existing_decisions():
    dates = ['20260518', '20260519', '20260520']
    records = [record('2026-05-19T08:00:00+09:00', 'a', .4)]
    before = morning_signals(records, dates)
    after = morning_signals(records + [record('2026-05-21T08:00:00+09:00', 'b', .8)], dates + ['20260521'])
    assert before[:-1] == after[:len(before)-1]


def test_morning_utc_conversion_and_prior_only_smoothing():
    from test_timefolio_heatmap import synthetic_panel
    p, ix = synthetic_panel(days=35); dates = ix['dates']
    cutoff = datetime.strptime(dates[1], '%Y%m%d').replace(hour=8, minute=55, tzinfo=KST)
    records = [record(cutoff.astimezone(timezone.utc).isoformat(), 'morning', .3),
               record((cutoff + timedelta(seconds=1)).isoformat(), 'too_late', .8)]
    records += [record(datetime.strptime(day, '%Y%m%d').replace(hour=8, tzinfo=KST).isoformat(), day, .5)
                for day in dates[2:]]
    rows = morning_signals(records, dates); schedule, available = schedules_from_records(p, dates, rows)
    assert available[0] and schedule['macro_stock'][0] == .3
    assert schedule['macro_smooth'][0] == .6 and schedule['macro_smooth'][1] == .3
    altered = deepcopy(rows)
    assert sum(r['available'] for r in altered[20:]) > 0
    for r in altered[20:]:
        if r['available']: r['stock_pct'] = .7
    later, _ = schedules_from_records(p, dates, altered)
    for name in schedule: np.testing.assert_array_equal(schedule[name][:20], later[name][:20])


def test_unsorted_or_duplicate_session_calendar_is_rejected():
    with pytest.raises(ValueError): morning_signals([], ['20260519', '20260518'])
    with pytest.raises(ValueError): morning_signals([], ['20260518', '20260518'])
