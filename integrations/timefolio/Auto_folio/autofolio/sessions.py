from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

KST = timezone(timedelta(hours=9))

SCHEDULE = {
    "KR_PRE_MARKET": (time(8, 0), time(8, 50)),
    "KR_TRADING": (time(9, 0), time(15, 30)),
    "KR_CLOSE_REVIEW": (time(15, 35), time(15, 50)),
    "KR_AFTER_MARKET": (time(15, 50), time(20, 0)),
    "US_TRADING": (time(22, 30), time(5, 0)),
}


def now_kst() -> datetime:
    return datetime.now(KST)


def _in_window(now: datetime, start: time, end: time) -> bool:
    t = now.timetz().replace(tzinfo=None)
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def current_session(now: datetime | None = None) -> str:
    now = now or now_kst()
    for name in ("KR_PRE_MARKET", "KR_TRADING", "KR_CLOSE_REVIEW", "KR_AFTER_MARKET", "US_TRADING"):
        if _in_window(now, *SCHEDULE[name]):
            return name
    return "OFF_HOURS"


def is_kr_session(session: str) -> bool:
    return session in {"KR_TRADING", "KR_PRE_MARKET", "KR_AFTER_MARKET", "KR_CLOSE_REVIEW"}


def is_kr_extended_hours(session: str) -> bool:
    return session in {"KR_PRE_MARKET", "KR_AFTER_MARKET"}


def kr_exchange_for_session(session: str) -> str:
    return "NXT" if is_kr_extended_hours(session) else "KRX"


def is_mock_kr_tradable(session: str) -> bool:
    """QuantInSight mock-account rule: mock accounts do not use NXT."""
    return session == "KR_TRADING"


def is_kr_code(code: object) -> bool:
    s = str(code or "").strip()
    return len(s) == 6 and s.isdigit()
