"""Daily rolling 36-month windows, pinned throughout each immutable trial."""
import calendar
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime,date,timedelta,timezone
from functools import lru_cache
from zoneinfo import ZoneInfo


_WINDOW=ContextVar('evaluation_window',default=None)


@contextmanager
def using_window(start,end):
    first=datetime.strptime(start,'%Y%m%d').date();last=datetime.strptime(end,'%Y%m%d').date()
    if first>last:raise ValueError('Invalid evaluation window')
    token=_WINDOW.set((start,end))
    try:yield (start,end)
    finally:_WINDOW.reset(token)


def window(today=None):
    if today is None and _WINDOW.get() is not None:return _WINDOW.get()
    today=today or datetime.now(ZoneInfo('Asia/Seoul')).date()
    end=today-timedelta(days=1)
    start=date(today.year-3,today.month,min(today.day,calendar.monthrange(today.year-3,today.month)[1]))
    return start.strftime('%Y%m%d'),end.strftime('%Y%m%d')


def _exchange_calendar(market,start,end):
    try:
        import exchange_calendars as calendars
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path.home()/'vault/QuantInSight/python-deps'))
        import exchange_calendars as calendars
    return calendars.get_calendar('XNYS' if market=='us' else 'XKRX',start=start,end=end)


@lru_cache(maxsize=16)
def expected_dates(market,start,end):
    if market=='crypto':
        first=datetime.strptime(start,'%Y%m%d').date();last=datetime.strptime(end,'%Y%m%d').date()
        return tuple((first+timedelta(days=i)).strftime('%Y%m%d') for i in range((last-first).days+1))
    cal=_exchange_calendar(market,start,end)
    # KRX announced these additional 2026 closures on 2026-05-20.
    # https://stock.mk.co.kr/news/disclosure/template/1029043 (exchange notice)
    closures={'20260603','20260717'} if market!='us' else set()
    return tuple(d.strftime('%Y%m%d') for d in cal.sessions if d.strftime('%Y%m%d') not in closures)


def last_complete_date(market,now=None):
    """Do not seal an intraday bar merely because KST has reached the next date."""
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('Completion clock needs a timezone')
    now=now.astimezone(timezone.utc)
    if market=='crypto':return (now.date()-timedelta(days=1)).strftime('%Y%m%d')
    start=(now.date()-timedelta(days=45)).strftime('%Y%m%d');end=now.strftime('%Y%m%d')
    cal=_exchange_calendar(market,start,end)
    for day in reversed(expected_dates(market,start,end)):
        if cal.session_close(day).to_pydatetime()<=now:return day
    raise ValueError('완료된 거래일을 확인할 수 없습니다.')


def accepts(summary,today=None):
    start,end=window(today)
    expected=expected_dates(summary.get('market','kr'),start,end)
    return (summary.get('months')==36 and summary.get('start')==expected[0] and summary.get('end')==expected[-1]
            and summary.get('sessions',len(expected))==len(expected))


def require_complete(dates,market,today=None):
    expected=expected_dates(market,*window(today))
    if tuple(dates)!=expected:raise ValueError('최근 36개월의 전체 거래일 자료가 필요합니다.')
    return expected
