from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class PriceType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Holding:
    code: str
    qty: int
    avg_price: float = 0.0
    cur_price: float = 0.0
    sellable_qty: int | None = None

    @property
    def pnl_pct(self) -> float:
        if self.avg_price <= 0 or self.cur_price <= 0:
            return 0.0
        return (self.cur_price / self.avg_price - 1.0) * 100.0

    @property
    def notional(self) -> float:
        return max(0, int(self.qty)) * max(0.0, float(self.cur_price or self.avg_price or 0.0))


@dataclass
class BuyingPower:
    cash: float
    total_eval: float
    pnl_ratio: float = 0.0
    ok: bool = True


@dataclass
class OrderDraft:
    ticker: str
    side: Literal["buy", "sell"]
    qty: int
    price_type: PriceType = PriceType.MARKET
    limit_price: float | None = None
    market: str = "KR"
    exchange: str = "KRX"
    reason: str = ""
    entry_mode: str | None = None
    entry_limit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "qty": self.qty,
            "price_type": self.price_type.value if isinstance(self.price_type, PriceType) else self.price_type,
            "limit_price": self.limit_price,
            "market": self.market,
            "exchange": self.exchange,
            "reason": self.reason,
            "entry_mode": self.entry_mode,
            "entry_limit": self.entry_limit,
            **self.metadata,
        }


@dataclass
class OrderResult:
    ticker: str
    side: str
    qty: int
    result: str
    accepted: bool = False
    filled: bool = False
    fill_price: float | None = None
    fill_note: str = ""
