"""Execution — 주문 인터페이스 (모의 기본, 실전 어댑터 스텁).

PaperBroker: 롤링 버퍼의 최신 종가 ± 슬리피지로 즉시 체결, trades.csv 기록.
KISBroker : 실전 스텁. LIVE_TRADING=True 로 켜기 전에 ArQuant 의
            infra/kis_broker.py (kr_buy/kr_sell, 6자리 코드) 를 어댑터로 연결할 것.
"""
import csv
import logging
from datetime import datetime

import config
from utils import market_time as mt
from utils.notify import notify

log = logging.getLogger("lag.executor")


class Broker:
    def buy_market(self, code, qty, ref_price):
        raise NotImplementedError

    def sell_market(self, code, qty, ref_price):
        raise NotImplementedError


class PaperBroker(Broker):
    """모의 체결: ref_price 에 슬리피지(bp)를 얹어 즉시 전량 체결."""

    def __init__(self, trades_path=None):
        self.trades_path = trades_path or config.TRADES_CSV
        if not self.trades_path.exists():
            with open(self.trades_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts", "side", "code", "qty", "price", "reason"])

    def _log_trade(self, side, code, qty, price, reason):
        with open(self.trades_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [mt.now_kst().isoformat(timespec="seconds"), side, code, qty,
                 f"{price:.1f}", reason])
        log.info("[PAPER %s] %s x%d @ %.1f (%s)", side, code, qty, price, reason)

    def buy_market(self, code, qty, ref_price, reason=""):
        fill = ref_price * (1 + config.PAPER_SLIPPAGE_BPS / 10_000)
        self._log_trade("BUY", code, qty, fill, reason)
        return fill

    def sell_market(self, code, qty, ref_price, reason=""):
        fill = ref_price * (1 - config.PAPER_SLIPPAGE_BPS / 10_000)
        self._log_trade("SELL", code, qty, fill, reason)
        return fill


class KISBroker(Broker):
    """실전 주문 어댑터 스텁.

    연결 방법(ArQuant 재사용):
        from infra.kis_broker import KISBroker as ArqKIS   # ArQuant 쪽
        kr_buy(code, qty) / kr_sell(code, qty)             # 국내 시장가
    실주문은 절대 조용히 누락하면 안 된다 — 전송 실패 시 notify + 재시도 폴백 필수.
    """

    def __init__(self):
        raise NotImplementedError(
            "KISBroker 는 아직 미연결. config.LIVE_TRADING=False(모의)로 운용하거나 "
            "ArQuant infra/kis_broker.py 어댑터를 구현할 것.")


def make_broker():
    if config.LIVE_TRADING:
        notify("LIVE_TRADING=True — KIS 실전 브로커 초기화 시도")
        return KISBroker()
    return PaperBroker()
