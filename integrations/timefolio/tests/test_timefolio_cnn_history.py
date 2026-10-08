import sqlite3
from pathlib import Path

import pytest

from quant.timefolio_cnn_history import aggregate_day, canonical_hhmm, connect, months, read_month


def bar(ts, price=100, volume=10):
    return (ts, price, price + 1, price - 1, price, volume)


def test_provider_alignment_and_non_nxt_auction():
    toss = [bar('20230102090600'), bar('20230102153000', 102, 100)]
    kis = [bar('20230102090500'), bar('20230102153000', 102, 100)]
    a, b = aggregate_day('000020', 'toss', toss), aggregate_day('000020', 'kis', kis)
    for key in ['open', 'high', 'low', 'close', 'volume', 'exec_price_proxy', 'exec_volume']:
        assert a[key] == b[key]
    assert a['has_1530_trade'] and a['krx_venue_verified']


def test_nxt_auction_and_venue_are_not_silently_krx():
    a = aggregate_day('005930', 'toss', [bar('20250401080100'), bar('20250401090600'),
        bar('20250401153000'), bar('20250401153100', 102, 100), bar('20250401200000')])
    assert a['has_1530_trade'] and a['canonical_duplicates'] == 0
    assert a['regular_bars'] == 3 and a['outside_bars'] == 2
    assert a['volume'] == 120 and not a['krx_venue_verified']


def test_no_zero_volume_invented_trade_or_afterhours_close():
    a = aggregate_day('000020', 'kis', [bar('20240103090000', 90, 0), bar('20240103090500'),
        bar('20240103153000', 105), bar('20240103180000', 900)])
    assert a['open'] == 100 and a['close'] == 105 and a['volume'] == 20
    assert not a['session_calendar_verified']


def test_bad_ohlc_and_late_session_remain_visible():
    a = aggregate_day('000020', 'kis', [('20231116090000', 100, 99, 101, 100, 10),
        bar('20231116100000'), bar('20231116163000', 110)])
    assert a['invalid_bars'] == 1 and a['outside_bars'] == 1
    assert a['late_open_candidate'] and not a['has_1530_trade']


def test_unordered_or_mixed_dates_rejected():
    with pytest.raises(ValueError):
        aggregate_day('000020', 'kis', [bar('20230102090100'), bar('20230102090000')])
    with pytest.raises(ValueError):
        aggregate_day('000020', 'kis', [bar('20230102090100'), bar('20230103090000')])


def test_invalid_clock_rejected():
    for ts in ['20230102096000', '20230102240000', '20230102090010']:
        with pytest.raises(ValueError):
            canonical_hhmm(ts, 'toss')


def test_source_is_read_only_and_cutoff_excludes_reserved_day(tmp_path):
    p = tmp_path / 'bars.db'
    with sqlite3.connect(p) as c:
        c.execute('CREATE TABLE bars(code TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL,PRIMARY KEY(code,ts))')
        c.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?)',
                      [('000020', *bar(day + '090000')) for day in ['20260923', '20260924', '20260928']])
    lo, hi = list(months('20260901', '20260923'))[0]
    rows = read_month(p, '000020', lo, hi)
    assert [r[0][:8] for r in rows] == ['20260923']
    with connect(p) as c:
        with pytest.raises(sqlite3.OperationalError):
            c.execute('DELETE FROM bars')


def test_month_bounds_cover_leap_day_without_overlap():
    result = list(months('20240210', '20240305'))
    assert result == [('20240210', '20240301'), ('20240301', '20240306')]
