"""QuantInSight KR mock-account trading constants.

The values here mirror the Korean-market subset from
QuantInSight의 한국 모의계좌 설정 가운데 필요한 값만 독립시킨 모듈입니다.
Auto_folio deliberately excludes US, derivatives and NXT for mock accounts.
"""

from __future__ import annotations

LIVE_TRADING = False
MAX_TRADES_PER_CYCLE = 2

ENABLE_SELL_REBALANCE = True
TAKE_PROFIT_PCT = 12.0
STOP_LOSS_PCT = 7.0
TRAILING_TAKE_PROFIT_PCT = 0.0
TRIM_OVER_RATIO = True
ALLOW_DAY_TRADING = True
MIN_HOLDING_DAYS_FOR_SELL = 0.5

# QuantInSight supports NXT for live accounts, but its README and KIS broker treat
# mock accounts as NXT-unsupported. Auto_folio is Timefolio/mock-first, so NXT
# sessions are hard-disabled and only KRX regular trading is executable.
ENABLE_NXT_EXTENDED_HOURS = False
ENABLE_NXT_PRE_MARKET = False
ENABLE_NXT_AFTER_MARKET = False
EXT_HOURS_LIMIT_SLIPPAGE_PCT = 0.5
EXT_HOURS_MAX_PREMIUM_PCT = 1.5

ALLOW_US_STOCKS = False
ALLOW_DERIVATIVES = False
ENABLE_CHEAP_FALLBACK = False

MAX_ORDER_QTY = 0
HARD_MAX_ORDER_QTY = 1000
PER_ORDER_BUDGET_RATIO = 0.10
MAX_CYCLE_BUDGET_RATIO = 0.25
MIN_CASH_BUFFER = 1.10
PER_ORDER_BUDGET_OVERSHOOT = 1.20

CONSERVATIVE_MDD = 0.05
CONSERVATIVE_STOCK_RATIO = 0.15

MIN_TRADABLE_CASH_KRW = 5000
