"""The last 36 completed calendar months, including every scheduled session."""
from datetime import datetime,date,timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo


def window(today=None):
    today=today or datetime.now(ZoneInfo('Asia/Seoul')).date()
    end=today.replace(day=1)-timedelta(days=1)
    return date(today.year-3,today.month,1).strftime('%Y%m%d'),end.strftime('%Y%m%d')


@lru_cache(maxsize=16)
def expected_dates(market,start,end):
    if market=='crypto':
        first=datetime.strptime(start,'%Y%m%d').date();last=datetime.strptime(end,'%Y%m%d').date()
        return tuple((first+timedelta(days=i)).strftime('%Y%m%d') for i in range((last-first).days+1))
    try:
        import exchange_calendars as calendars
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path.home()/'vault/QuantInSight/python-deps'))
        import exchange_calendars as calendars
    cal=calendars.get_calendar('XNYS' if market=='us' else 'XKRX',start=start,end=end)
    return tuple(d.strftime('%Y%m%d') for d in cal.sessions)


def accepts(summary,today=None):
    start,end=window(today)
    expected=expected_dates(summary.get('market','kr'),start,end)
    return (summary.get('months')==36 and summary.get('start')==expected[0] and summary.get('end')==expected[-1]
            and summary.get('sessions',len(expected))==len(expected))


def require_complete(dates,market,today=None):
    expected=expected_dates(market,*window(today))
    if tuple(dates)!=expected:raise ValueError('최근 36개월의 전체 거래일 자료가 필요합니다.')
    return expected
