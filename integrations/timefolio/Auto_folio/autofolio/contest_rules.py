from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# AUTOFOLIO_RELAX_SECTOR=1 이면 섹터 데이터 부재를 위반이 아닌 경고로 처리한다.
# 네이버 크롤링으로는 GICS 섹터·시장 섹터 비중을 구할 수 없어, 무인 페이퍼 러너(ArcTrade)가
# 전 종목 매수 거절되는 것을 막기 위한 완화 — 데이터가 있으면 섹터 한도는 그대로 검증한다.
# (QuantInSight 본체는 이 env 를 켜지 않아 대회 룰 그대로 엄격 검증.)
_RELAX_SECTOR = os.environ.get("AUTOFOLIO_RELAX_SECTOR", "") == "1"

MIN_AVG_5D_TRADING_VALUE_KRW = 3_000_000_000
MIN_MARKET_CAP_KRW = 100_000_000_000
SMALL_CAP_THRESHOLD_KRW = 1_000_000_000_000
SMALL_CAP_TOTAL_LIMIT = 0.30
BASE_POSITION_LIMIT = 0.15
POSITION_LIMIT_EXCEPTIONS = {
    "005930": 0.40,  # Samsung Electronics
    "000660": 0.30,  # SK hynix, effective 2026-07-01
}
ALLOWED_MARKETS = {"KOSPI", "KOSDAQ"}
BLOCKED_FLAG_WORDS = ("투자주의", "투자경고", "투자위험", "투자주의환기", "관리", "거래정지")


@dataclass
class RuleViolation:
    code: str
    message: str


@dataclass
class RuleCheckResult:
    ok: bool
    violations: list[RuleViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, code: str, message: str) -> None:
        self.violations.append(RuleViolation(code, message))
        self.ok = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [v.__dict__ for v in self.violations],
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def _f(v: Any, default: float | None = 0.0) -> float | None:
    if v is None or v == "":
        return default
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).replace(",", "")))
    except Exception:
        return default


def normalize_meta(ticker: str, meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = dict(meta or {})
    ticker = str(ticker or meta.get("ticker") or "").strip().zfill(6)
    flags = meta.get("flags") or []
    if isinstance(flags, str):
        flags = [x.strip() for x in flags.split(",") if x.strip()]
    out = {
        "ticker": ticker,
        "name": str(meta.get("name") or "").strip(),
        "market": str(meta.get("market") or "").strip().upper(),
        "is_common_stock": meta.get("is_common_stock"),
        "listed_business_days": _i(meta.get("listed_business_days"), -1),
        "avg_5d_trading_value_krw": _f(meta.get("avg_5d_trading_value_krw"), None),
        "market_cap_krw": _f(meta.get("market_cap_krw"), None),
        "sector": str(meta.get("sector") or "").strip(),
        "market_sector_weight_pct": _f(meta.get("market_sector_weight_pct"), None),
        "flags": flags,
        "last_price": _f(meta.get("last_price"), None),
        "updated_at": meta.get("updated_at") or datetime.now(timezone.utc).isoformat(),
    }
    return out


def position_limit(ticker: str) -> float:
    return POSITION_LIMIT_EXCEPTIONS.get(str(ticker).zfill(6), BASE_POSITION_LIMIT)


def sector_limit(meta: dict[str, Any]) -> float | None:
    w = _f(meta.get("market_sector_weight_pct"), None)
    if w is None:
        return None
    return 0.10 if w <= 5.0 else (w * 2.0 / 100.0)


def validate_order(account: dict[str, Any], side: Literal["buy", "sell"], ticker: str, qty: int,
                   price: float, metas: dict[str, dict[str, Any]], *,
                   relax_sector: bool | None = None) -> RuleCheckResult:
    # relax_sector: None=env 기본(_RELAX_SECTOR), True/False=호출자 명시 오버라이드.
    # 스웜(TimefolioBroker) 경로는 섹터 데이터를 자동 수급할 수 없어 True 로 부른다.
    relax = _RELAX_SECTOR if relax_sector is None else bool(relax_sector)
    ticker = str(ticker).zfill(6)
    side = "buy" if side == "buy" else "sell"
    qty = _i(qty)
    price = float(price or 0)
    result = RuleCheckResult(ok=True)
    if qty <= 0:
        result.add("bad_qty", "주문 수량은 1주 이상이어야 합니다.")
        return result
    if price <= 0:
        result.add("bad_price", "주문 가격을 확인할 수 없습니다.")
        return result

    positions = dict((account or {}).get("positions") or {})
    cash = float((account or {}).get("cash") or 0.0)
    before_qty = _i((positions.get(ticker) or {}).get("qty"))
    if side == "sell":
        if before_qty < qty:
            result.add("sell_qty", f"보유 수량 부족: 보유 {before_qty}주, 매도 {qty}주")
        return result

    meta = normalize_meta(ticker, metas.get(ticker))
    _validate_buy_universe(result, meta, relax)

    order_amount = qty * price
    if cash < order_amount:
        result.add("cash", f"현금 부족: 주문 {order_amount:,.0f}원, 현금 {cash:,.0f}원")

    projected = _project_positions(positions, ticker, qty, price, metas)
    cash_after = cash - order_amount
    total_after = max(0.0, cash_after) + sum(max(0, _i(p.get("qty"))) * float(p.get("last_price") or p.get("avg_price") or 0) for p in projected.values())
    if total_after <= 0:
        result.add("portfolio_value", "포트폴리오 평가금액을 계산할 수 없습니다.")
        return result

    pos_value = _i(projected[ticker].get("qty")) * price
    pos_weight = pos_value / total_after
    max_pos = position_limit(ticker)
    result.metrics["position_weight_pct"] = pos_weight * 100.0
    result.metrics["position_limit_pct"] = max_pos * 100.0
    if pos_weight > max_pos + 1e-9:
        result.add("position_limit", f"종목별 편입 한도 초과: {pos_weight*100:.2f}% > {max_pos*100:.0f}%")

    _validate_sector_limit(result, projected, metas, total_after, meta, relax)
    _validate_small_cap_limit(result, projected, metas, total_after)
    return result


def _validate_buy_universe(result: RuleCheckResult, meta: dict[str, Any], relax: bool = False) -> None:
    ticker = meta.get("ticker") or ""
    if meta.get("market") not in ALLOWED_MARKETS:
        result.add("market", f"{ticker} 매수 불가: KOSPI/KOSDAQ 보통주만 가능합니다.")
    if meta.get("is_common_stock") is not True:
        result.add("common_stock", f"{ticker} 매수 불가: 보통주 여부가 확인되지 않았습니다.")
    if meta.get("listed_business_days", -1) < 5:
        result.add("listed_days", "최근 거래일 5일 미만 또는 상장 6영업일 미도래 종목입니다.")
    avg5 = _f(meta.get("avg_5d_trading_value_krw"), None)
    if avg5 is None:
        result.add("avg5_missing", "5일 평균 거래대금 데이터가 없습니다.")
    elif avg5 <= MIN_AVG_5D_TRADING_VALUE_KRW:
        result.add("avg5", f"5일 평균 거래대금 30억 이하: {avg5:,.0f}원")
    cap = _f(meta.get("market_cap_krw"), None)
    if cap is None:
        result.add("market_cap_missing", "시가총액 데이터가 없습니다.")
    elif cap < MIN_MARKET_CAP_KRW:
        result.add("market_cap", f"시가총액 1,000억원 미만: {cap:,.0f}원")
    flags = [str(x) for x in (meta.get("flags") or [])]
    bad_flags = [x for x in flags if any(word in x for word in BLOCKED_FLAG_WORDS)]
    if bad_flags:
        result.add("blocked_flags", "매수 불가 지정 상태: " + ", ".join(bad_flags))
    if not meta.get("sector"):
        if relax:
            result.warnings.append("GICS 섹터 데이터 없음 — 섹터 한도 검증 생략")
        else:
            result.add("sector_missing", "GICS 섹터 데이터가 없습니다.")
    if _f(meta.get("market_sector_weight_pct"), None) is None:
        if relax:
            result.warnings.append("시장 섹터 비중 데이터 없음 — 섹터 한도 검증 생략")
        else:
            result.add("sector_weight_missing", "시장 섹터 비중 데이터가 없습니다.")


def _project_positions(positions: dict[str, Any], ticker: str, buy_qty: int, price: float,
                       metas: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for code, pos in positions.items():
        code6 = str(code).zfill(6)
        meta = normalize_meta(code6, metas.get(code6))
        last = _f(meta.get("last_price"), None) or _f(pos.get("last_price"), None) or _f(pos.get("avg_price"), 0.0)
        out[code6] = {"qty": _i(pos.get("qty")), "avg_price": _f(pos.get("avg_price"), 0.0), "last_price": last}
    cur = out.get(ticker, {"qty": 0, "avg_price": price, "last_price": price})
    old_qty = _i(cur.get("qty"))
    old_avg = _f(cur.get("avg_price"), price) or price
    new_qty = old_qty + buy_qty
    cur["qty"] = new_qty
    cur["avg_price"] = ((old_avg * old_qty) + (price * buy_qty)) / max(1, new_qty)
    cur["last_price"] = price
    out[ticker] = cur
    return out


def _validate_sector_limit(result: RuleCheckResult, positions: dict[str, Any], metas: dict[str, dict[str, Any]],
                           total_after: float, target_meta: dict[str, Any], relax: bool = False) -> None:
    target_sector = target_meta.get("sector") or ""
    limit = sector_limit(target_meta)
    if not target_sector or limit is None:
        return
    value = 0.0
    missing = []
    for code, pos in positions.items():
        meta = normalize_meta(code, metas.get(code))
        if not meta.get("sector"):
            missing.append(code)
            continue
        if meta.get("sector") == target_sector:
            value += _i(pos.get("qty")) * float(pos.get("last_price") or pos.get("avg_price") or 0)
    if missing:
        if relax:
            result.warnings.append("보유 종목 섹터 데이터 없음(검증 제외): " + ", ".join(missing))
        else:
            result.add("held_sector_missing", "보유 종목 섹터 데이터가 없습니다: " + ", ".join(missing))
            return
    weight = value / total_after
    result.metrics["sector_weight_pct"] = weight * 100.0
    result.metrics["sector_limit_pct"] = limit * 100.0
    if weight > limit + 1e-9:
        result.add("sector_limit", f"섹터 편입 한도 초과: {weight*100:.2f}% > {limit*100:.2f}%")


def _validate_small_cap_limit(result: RuleCheckResult, positions: dict[str, Any], metas: dict[str, dict[str, Any]], total_after: float) -> None:
    value = 0.0
    missing = []
    for code, pos in positions.items():
        meta = normalize_meta(code, metas.get(code))
        cap = _f(meta.get("market_cap_krw"), None)
        if cap is None:
            missing.append(code)
            continue
        if cap < SMALL_CAP_THRESHOLD_KRW:
            value += _i(pos.get("qty")) * float(pos.get("last_price") or pos.get("avg_price") or 0)
    if missing:
        result.add("held_market_cap_missing", "보유 종목 시가총액 데이터가 없습니다: " + ", ".join(missing))
        return
    weight = value / total_after
    result.metrics["small_cap_weight_pct"] = weight * 100.0
    result.metrics["small_cap_limit_pct"] = SMALL_CAP_TOTAL_LIMIT * 100.0
    if weight > SMALL_CAP_TOTAL_LIMIT + 1e-9:
        result.add("small_cap_limit", f"시총 1조 미만 종목 합산 한도 초과: {weight*100:.2f}% > 30.00%")
