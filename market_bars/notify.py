"""Fail-safe 알림 — 크롤러가 연속 실패해도 시스템을 죽이지 않고 로깅만 한다.

구 Lag_Trading 의 텔레그램 배관은 이관에서 제외(QuantInSight 는 자체 notifier 사용).
크롤러 이상은 로그로만 남긴다 — 알림 실패가 수집 루프를 죽이면 안 된다.
"""
import logging

log = logging.getLogger("quantinsight.bars")


def notify(message, level=logging.WARNING):
    log.log(level, "[BAR-CRAWLER ALERT] %s", message)
