"""주문창(in_order_session) 게이트 — 수집창보다 좁아야 한다.

TMS 는 15:31 부터 전건 "주문 시작 시간이 폐장 이후" 로 거부한다(2026-07-29 실측).
수집창(~15:35)으로 주문을 게이트하면 폐장 후 신호마다 브라우저 왕복 후 거부되어 로그만 오염된다.

python3.12 -m pytest tests/test_order_session.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import market_time as mt              # noqa: E402


def _at(h, m, day=29):          # 2026-07-29 = 수요일
    return datetime(2026, 7, day, h, m, tzinfo=mt.KST)


def test_order_window_is_narrower_than_crawl_window():
    assert mt.in_order_session(_at(9, 0))        # 개장
    assert mt.in_order_session(_at(15, 29))      # 종가 동시호가 — 아직 접수됨
    assert not mt.in_order_session(_at(15, 30))  # 여기부터 TMS 거부
    assert not mt.in_order_session(_at(15, 34))
    assert not mt.in_order_session(_at(8, 59))
    # 수집은 마감 프린트까지 계속되어야 한다 — 두 창이 갈라지는 게 이 수정의 핵심
    assert mt.in_crawl_session(_at(15, 34))


def test_weekend_blocked():
    assert not mt.in_order_session(_at(11, 0, day=25))   # 2026-07-25 토요일
