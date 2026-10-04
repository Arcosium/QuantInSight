"""에이전트 성과 귀인 — 순수 함수 (사장 지시 2026-06-04 ④).
IO 없음: 신호/체결/자산/가격을 인자로 받아 에이전트별 예측력 지표를 계산한다.
- information_coefficient: 예측(점수·감성) vs 후속수익 스피어만 순위상관(-1..1).
- slippage_stats: 결정가 대비 체결가 불리도(bps, +면 불리).
- alpha_beta: 포트 수익 vs 벤치마크 OLS(beta·alpha).
- compute_scorecard: 위를 조립해 {quant, news, slippage, portfolio, ...} dict.
표본 부족(3 미만)·결측은 None/n 으로 정직히 표기(무음 금지)."""
from typing import Callable, Dict, List, Optional, Tuple


def _rank(xs: List[float]) -> List[float]:
    """평균 순위(동점은 평균). 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """피어슨 상관 (3쌍 미만·무분산이면 None). 프로필 수익률 상관 등 외부 호출용 공개 래퍼."""
    return _pearson(xs, ys)


def information_coefficient(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """[(signal, forward_return)] → 스피어만 순위상관. 3쌍 미만이면 None."""
    pairs = [(float(s), float(r)) for s, r in pairs if s is not None and r is not None]
    if len(pairs) < 3:
        return None
    sigs = [p[0] for p in pairs]; rets = [p[1] for p in pairs]
    return _pearson(_rank(sigs), _rank(rets))


def confidence_from_ic(ic, n, *, min_n: int = 20, full_n: int = 100,
                       scale: float = 2.5, floor: float = 0.1, cap: float = 1.0) -> float:
    """에이전트 예측력(IC) → 0~1 확신도 (2026-06-15 ROI#2). 블랙-리터만 Ω·사이징 틸트의 입력.
    표본(n)이 min_n 미만이거나 IC 없음 → 중립 0.5(과신 방지). IC 음수(역사적 오답) → 0.5 미만.
    표본이 min_n~full_n 사이면 중립 쪽으로 축소(작은 표본 과신 방지)."""
    if ic is None or n is None or n < min_n:
        return 0.5
    raw = 0.5 + float(ic) * scale
    conf = max(floor, min(cap, raw))
    w = max(0.0, min(1.0, (n - min_n) / max(1, full_n - min_n)))  # 표본 신뢰 가중
    return round(0.5 + (conf - 0.5) * w, 3)


def quant_confidence(uid, *, window_days: int = 30, max_signals: int = 300):
    """라이브 스코어카드에서 퀀트 신호의 IC 를 산출해 0~1 확신도로 (2026-06-15 ROI#2 배선).
    반환 (confidence, ic, n). 표본/데이터 부족이면 (0.5, None, 0). 베스트에포트(예외 시 중립)."""
    try:
        from infra import scorecard_store
        from tools.market_data import forward_return_after
        sigs = scorecard_store.list_signals(uid=uid, limit=max_signals)
        pairs = []
        for s in (sigs or []):
            if s.get("quant_score") is None:
                continue
            fwd = forward_return_after(s.get("code"), s.get("ts"), window_days)
            if fwd is None:
                continue
            pairs.append((float(s["quant_score"]), float(fwd)))
        ic = information_coefficient(pairs)
        return confidence_from_ic(ic, len(pairs)), ic, len(pairs)
    except Exception:
        return 0.5, None, 0


def slippage_stats(fills: List[Dict]) -> Dict:
    """fills: [{side, decision_price, fill_price}] → {mean_bps, n}. +bps = 불리(매수 비싸게/매도 싸게)."""
    bps = []
    for f in (fills or []):
        dp = f.get("decision_price"); fp = f.get("fill_price")
        if not dp or not fp or float(dp) <= 0:
            continue
        diff = (float(fp) - float(dp)) / float(dp)
        if str(f.get("side", "")).lower().startswith("sell"):
            diff = -diff                      # 매도는 싸게 체결될수록 불리
        bps.append(diff * 10000.0)
    return {"mean_bps": (sum(bps) / len(bps)) if bps else None, "n": len(bps)}


def alpha_beta(port_returns: List[float], bench_returns: List[float]) -> Optional[Dict]:
    """포트/벤치 수익률 시계열 OLS. 3점 미만/벤치 무분산이면 None. {alpha, beta, n}."""
    n = min(len(port_returns or []), len(bench_returns or []))
    if n < 3:
        return None
    p = [float(x) for x in port_returns[:n]]; b = [float(x) for x in bench_returns[:n]]
    mb = sum(b) / n; mp = sum(p) / n
    vb = sum((x - mb) ** 2 for x in b)
    if vb <= 0:
        return None
    beta = sum((b[i] - mb) * (p[i] - mp) for i in range(n)) / vb
    return {"alpha": mp - beta * mb, "beta": beta, "n": n}


def compute_scorecard(signals: List[Dict], trades: List[Dict], equity: List[Dict],
                      bench: List[float], *, price_lookup: Callable[[str, str], Optional[float]],
                      window_days: int = 30) -> Dict:
    """에이전트별 지표 조립. price_lookup(code, signal_ts) → 후속수익률(없으면 None=표본제외).
    quant/news IC 는 signals, slippage 는 trades, portfolio 알파/베타는 equity vs bench."""
    q_pairs, n_pairs = [], []
    for s in (signals or []):
        fwd = price_lookup(s.get("code"), s.get("ts"))
        if fwd is None:
            continue
        if s.get("quant_score") is not None:
            q_pairs.append((float(s["quant_score"]), float(fwd)))
        if s.get("news_sentiment") is not None:
            n_pairs.append((float(s["news_sentiment"]), float(fwd)))
    # 포트 수익률 시계열 — equity[{total_eval}] 연속 차분
    evals = [float(e["total_eval"]) for e in (equity or []) if e.get("total_eval")]
    port_rets = [(evals[i] / evals[i - 1] - 1.0) for i in range(1, len(evals)) if evals[i - 1] > 0]
    return {
        "quant": {"ic": information_coefficient(q_pairs), "n": len(q_pairs)},
        "news": {"ic": information_coefficient(n_pairs), "n": len(n_pairs)},
        "slippage": slippage_stats(trades or []),
        "portfolio": alpha_beta(port_rets, bench or []),
        "window_days": window_days,
        "signal_count": len(signals or []),
    }
