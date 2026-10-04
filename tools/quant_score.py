"""결정론 점수 엔진 — 순수 함수 (사장 지시 2026-06-04: LLM 일관성 문제 → 점수는 무조건 Python).
spec: docs/superpowers/specs/2026-06-04-deterministic-score-engine-design.md

두 계층 모두 스케일 안정 정규화 `norm(w,s)=5+5·Σ(w·s)/Σ|w|` 를 쓴다 — 가중치 크기 무관,
부호·비율만 의미, 결과 0~10 고정. IO·LLM 없음(전부 순수 함수).
"""
from typing import Dict, Optional, Tuple, Any


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _norm(weights: Dict[str, float], sigs: Dict[str, float]) -> float:
    """존재하는 신호(weights∩sigs)만 가중합 → 0~10. Σ|w|=0 또는 교집합 없음 → 중립 5.0."""
    num = 0.0
    den = 0.0
    for k, s in sigs.items():
        w = weights.get(k)
        if w is None:
            continue
        num += float(w) * float(s)
        den += abs(float(w))
    if den <= 0:
        return 5.0
    return _clamp(5.0 + 5.0 * num / den, 0.0, 10.0)


def indicator_signals(ind: Dict[str, Any]) -> Dict[str, float]:
    """지표 raw dict → 신호 s∈[-1,+1] (+1=매수 우호). 원천 값이 없는 지표는 결과에서 생략."""
    s: Dict[str, float] = {}
    if ind.get("rsi14") is not None:
        s["rsi"] = _clamp((50.0 - float(ind["rsi14"])) / 30.0)
    if ind.get("macd_hist_pct") is not None:
        s["macd"] = _clamp(float(ind["macd_hist_pct"]) / 2.0)
    if ind.get("adx") is not None:
        mag = _clamp((float(ind["adx"]) - 20.0) / 20.0, 0.0, 1.0)
        d = 1.0 if float(ind.get("adx_dir", 1) or 1) >= 0 else -1.0
        s["adx"] = mag * d
    if ind.get("vwap_dev") is not None:
        s["vwap"] = _clamp(-float(ind["vwap_dev"]) / 15.0)
    if ind.get("sigma20") is not None:
        s["vol"] = _clamp((40.0 - float(ind["sigma20"])) / 40.0)
    if ind.get("mom_1m") is not None or ind.get("mom_3m") is not None:
        m1 = float(ind.get("mom_1m") or 0.0)
        m3 = float(ind.get("mom_3m") or 0.0)
        s["mom"] = _clamp((0.4 * m1 + 0.6 * m3) / 20.0)
    if ind.get("cmf") is not None:
        s["cmf"] = _clamp(float(ind["cmf"]))          # CMF 는 이미 -1..1
    if ind.get("flow") is not None:
        s["flow"] = _clamp(float(ind["flow"]))        # 사전 정규화 수급 신호
    if ind.get("high52_prox") is not None:
        s["high52"] = _clamp((float(ind["high52_prox"]) - 0.5) / 0.5)
    if ind.get("leadlag") is not None:
        s["leadlag"] = _clamp(float(ind["leadlag"]))   # 선행-후행 신호는 이미 -1..1 (사장 지시 2026-07-21)
    if ind.get("vol_surge") is not None:
        s["vol_surge"] = _clamp(float(ind["vol_surge"]))   # 거래량 급증: +100%(2배)→+1 (사장 지시 2026-07-21)
    return s


def compute_indicator_score(ind: Dict[str, Any], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """지표 신호 가중합 → (0~10, breakdown). weights 는 신호명(rsi/macd/.../high52) 키."""
    sigs = indicator_signals(ind)
    score = _norm(weights, sigs)
    breakdown = {k: float(weights.get(k, 0.0)) * v for k, v in sigs.items()}
    return score, breakdown


def news_score(sentiment: Optional[float]) -> Optional[float]:
    """뉴스 감성(-1..1) → 0~10. None(파싱 실패) → None(차원 제외 폴백)."""
    if sentiment is None:
        return None
    return _clamp(5.0 + 5.0 * _clamp(float(sentiment)), 0.0, 10.0)


def macro_score(stock_pct: Optional[float]) -> Optional[float]:
    """매크로 권고 주식%(0~100) → 0~10 (15%=중립 5점). None → None."""
    if stock_pct is None:
        return None
    return _clamp(5.0 + 5.0 * _clamp((float(stock_pct) - 15.0) / 25.0), 0.0, 10.0)


# ── 알파 귀인 태그 (P2, 사장 지시 2026-08-02) ────────────────────────────────
# "이 매수는 어느 알파에서 나왔나" 를 남기지 않으면 알파별 성과·상관을 영원히 알 수 없다.
# 점수 breakdown 에 이미 축별 기여도가 있으므로 LLM 호출 없이 결정론으로 태깅한다.
ALPHA_FAMILY = {
    "mom": "모멘텀", "high52": "모멘텀", "macd": "모멘텀",
    "adx": "추세",
    "rsi": "평균회귀", "vwap": "평균회귀",
    "vol": "저변동",
    "cmf": "수급", "flow": "수급",
    "vol_surge": "거래량이벤트", "leadlag": "선행-후행",
}


def alpha_tag(breakdown: Optional[Dict[str, Any]]) -> str:
    """점수 breakdown → 이 진입을 끌어올린 알파 계열 1개.

    breakdown 은 assemble_quant_score 반환값({S_quant,S_news,S_macro,indicators}).
    뉴스 점수가 강하고(≥7.5) 퀀트보다 우위면 '뉴스이벤트', 아니면 기여도 최대 지표 축의 계열.
    가점 축이 하나도 없으면(전부 감점) '기타' — 억지 분류 금지."""
    bd = breakdown or {}
    ind = bd.get("indicators") or {}
    top = max(((k, float(v)) for k, v in ind.items() if float(v) > 0),
              key=lambda kv: kv[1], default=None)
    news, quant = bd.get("S_news"), bd.get("S_quant")
    if news is not None and float(news) >= 7.5 and (quant is None or float(news) > float(quant)):
        return "뉴스이벤트"
    return ALPHA_FAMILY.get(top[0], top[0]) if top else "기타"


def combine_dimensions(scores: Dict[str, Optional[float]], dw: Dict[str, float]) -> float:
    """차원 점수(QUANT/NEWS/MACRO, 각 0~10 또는 None)를 차원 가중치 dw로 합성 → 0~10.
    None 차원은 제외, 전부 None 이면 중립 5.0. DW 음수면 그 차원 반전."""
    centered = {k: (float(v) - 5.0) / 5.0 for k, v in scores.items() if v is not None}
    return _norm(dw, centered)
