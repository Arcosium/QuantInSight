"""타임폴리오 대회 1주문 비중 상한 — 섹터 한도 위반 방지 (사장 지시 2026-07-08).

대회 룰: 섹터 비중 ≤ max(섹터 시장비중 2배, 10%). 대부분 섹터는 10%가 하한이라, 종목을 10%로
편입하면 상승 시 그 섹터가 10%를 넘어 가이드라인 위반이 난다. 그래서 일반 종목은 1주문을 9%로
제한해 상승 여유(1%)를 둔다.

단 삼성전자·SK하이닉스처럼 섹터(IT/반도체) 시장비중이 커서 섹터 한도가 크게 열려 있는 대형주는
9% 제한이 불필요하므로 제외하고, 종목 한도(15%) 아래까지 더 큰 비중을 허용한다.

※ '추가 편입 가능 비중'은 사이트에서 매도 미체결을 선반영하지 않으므로, 미체결 매도로 생길 여유는
   신뢰하지 않는다(이 상한은 목표 비중 자체를 낮춰 그 영향과 무관하게 여유를 확보한다).
"""
from __future__ import annotations

# 9% 제한에서 제외할(섹터 여유가 큰) 대형주 — 필요 시 추가.
SECTOR_EXEMPT_TICKERS = {
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
}

DEFAULT_MAX_ORDER_WEIGHT_PCT = 9.0    # 일반 종목: 1주문 최대 비중(총평가 대비 %)
EXEMPT_MAX_ORDER_WEIGHT_PCT = 14.0    # 제외 종목: 종목 한도 15% 아래 여유


def max_order_weight_pct(ticker: str) -> float:
    """이 종목을 1주문에 편입할 수 있는 최대 비중(%). 제외 대형주는 더 크게 허용."""
    t = str(ticker or "").strip().zfill(6)
    return EXEMPT_MAX_ORDER_WEIGHT_PCT if t in SECTOR_EXEMPT_TICKERS else DEFAULT_MAX_ORDER_WEIGHT_PCT


def max_order_qty(ticker: str, price: float, total_eval: float) -> int:
    """총평가·현재가 기준, 비중 상한을 넘지 않는 최대 매수 수량. 계산 불가 시 0."""
    try:
        price = float(price); total_eval = float(total_eval)
    except (TypeError, ValueError):
        return 0
    if price <= 0 or total_eval <= 0:
        return 0
    return int(max_order_weight_pct(ticker) / 100.0 * total_eval / price)
