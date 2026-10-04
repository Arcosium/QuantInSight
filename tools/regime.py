"""레짐 신호 — LLM 매크로 판단을 수치화 (2026-06-15 ROI#3).

사장 결정(2026-06-15): 시장 국면 '판단'은 LLM(글로벌리서치팀장)이 한다 — 예외 상황에 유연하게
대응해야 하므로 결정론 분류기보다 LLM이 낫다. 이 모듈은 그 LLM 권고(자산배분 %)를 켈리 분수·
주식 비중 상한이 먹을 수 있는 0~1 de-risk score 로 '변환만' 한다(판단 X). 파싱 실패 시 중립.
"""
from __future__ import annotations
from typing import Dict, Optional

from infra.asset_sleeves import parse_macro_sleeve_pct


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def regime_score_from_macro(macro_report: Optional[str]) -> Dict:
    """LLM 매크로 자산배분 권고 → {regime, score, cash_pct, stock_pct}.
    score(0~1) = 권고 현금비중(방어 강도). 현금 미언급이면 1-주식비중으로 근사. 둘 다 없으면 중립 0.5.
    regime: score<0.4 risk_on · 0.4~0.6 neutral · >0.6 risk_off."""
    stock = parse_macro_sleeve_pct(macro_report, "주식")
    cash = parse_macro_sleeve_pct(macro_report, "현금")
    if cash is not None:
        score = _clamp01(cash)
    elif stock is not None:
        score = _clamp01(1.0 - stock)
    else:
        return {"regime": "neutral", "score": 0.5, "cash_pct": None, "stock_pct": None}
    regime = "risk_off" if score > 0.6 else ("risk_on" if score < 0.4 else "neutral")
    return {"regime": regime, "score": round(score, 3),
            "cash_pct": cash, "stock_pct": stock}
