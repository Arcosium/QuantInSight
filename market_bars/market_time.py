"""한국 주식시장 세션/동시호가 시간 판정 (KST 기준).

동시호가(08:30~09:00 개장 전, 15:20~15:30 폐장) 구간은 연산 윈도우에서
완전히 제외한다 — 노이즈 격리 (스펙 §3).
정규 연산 대상 분봉: 09:01 ~ 15:19 (분봉 ts 는 해당 분의 '종료' 시각).
"""
from datetime import datetime, timedelta, timezone, time as dtime

KST = timezone(timedelta(hours=9))

# 정규 세션에서 '연산에 쓰는' 분봉 시각 범위 (그 분의 종료 시각 기준)
REGULAR_FIRST = dtime(9, 1)
REGULAR_LAST = dtime(15, 19)
# 폐장 동시호가 체결(15:30 프린트)은 수집은 하되 연산에서 제외
AUCTION_WINDOWS = (((dtime(8, 30), dtime(9, 0))), ((dtime(15, 20), dtime(15, 30))))


def now_kst():
    return datetime.now(KST)


def is_weekday(d):
    return d.weekday() < 5


def in_crawl_session(ts=None):
    """수집을 돌릴 시간대인가 — 평일 09:00~15:35 (마감 프린트 여유 포함)."""
    ts = ts or now_kst()
    if not is_weekday(ts):
        return False
    return dtime(9, 0) <= ts.time() <= dtime(15, 35)


def recent_trading_days(n, today=None):
    """오늘 포함 최근 n 영업일(주말 제외; 공휴일은 데이터 유무로 자연 필터됨)."""
    d = (today or now_kst()).date()
    days = []
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


