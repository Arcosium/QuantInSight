"""계정 주식 평가비중 계산 — 매크로 매수게이트용.

핵심(2026-06-05 버그수정): 해외 USD '예수금'은 주식이 아니므로 주식비중에서 제외한다.
기존 `(total_eval − KR현금)/total_eval` 은 해외 USD 예수금을 전부 주식으로 오분류해
모의계정(USD 예수금 385M·실제 주식 0)을 79% 주식비중으로 계산 → 매크로 게이트가 매 사이클
매수를 영구 차단(후보 0 동결)했다.
"""
from __future__ import annotations
from typing import Optional


def compute_stock_weight(total_eval: Optional[float], kr_cash: Optional[float],
                         total_eval_kr: Optional[float], overseas_stock_krw: Optional[float]) -> float:
    """실제 주식가치 / 총평가.

    주식가치 = (KR 총평가 − KR 현금)  +  해외 주식분(원화환산). 범위 [0,1].
    total_eval≤0 이면 0.0 (fail-open — 잘못된 차단으로 거래가 전면중단되는 것 방지)."""
    te = float(total_eval or 0.0)
    if te <= 0:
        return 0.0
    tek = float(total_eval_kr or 0.0) or te   # KR 총평가 없으면 total 로 폴백(해외 미보유 가정, 보수)
    kr_stock = max(0.0, tek - float(kr_cash or 0.0))
    os_stock = max(0.0, float(overseas_stock_krw or 0.0))
    stock = kr_stock + os_stock
    # 사장 지시 2026-06-16: 주식가치는 (총평가 − KR예수금)을 넘을 수 없다 — 현금은 주식이 아니다.
    # 모의계정 해외평가 환율 오염(_overseas_stock_krw 과대, exrt 비정상)으로 주식비중이 100% 로
    # 캡되어 매크로 게이트가 매수를 영구 차단하던 버그(uid2: 현금 59% 인데 주식 100% 오판) 방어.
    # 입력(total_eval_kr/overseas) 신뢰도와 무관하게 물리적 상한을 보장 — 실거래도 안전(현금은 비주식).
    stock = min(stock, max(0.0, te - float(kr_cash or 0.0)))
    return max(0.0, min(1.0, stock / te))
