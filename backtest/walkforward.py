"""워크포워드 백테스트 — 롤링 아웃오브샘플 성과 안정성 (2026-06-15 ROI#1).

⚠️ 정직성 경계: backtest/engine.py 와 같은 고정 모멘텀 프록시(SMA 돌파) + 결정론 규칙만 쓴다
(종목선정 능력 평가 아님). 여기서 더하는 가치는 **'한 시기의 운'을 걸러내는 분포 측정**이다:
히스토리를 연속 윈도우로 쪼개 각 구간을 평가하면, 단일 백테스트의 한 숫자가 특정 구간 의존인지
(과적합 증상) vs 구간 전반에 견고한지가 드러난다. 운용지원실장의 토요일 튜닝 입력으로 쓴다.
"""
from __future__ import annotations
import math
from typing import Dict, List

from backtest.engine import run_backtest


def _aggregate(windows: List[Dict]) -> Dict:
    """윈도우별 성과 → 집계 분포. 일관성(pct_positive)·최악 구간이 핵심."""
    n = len(windows)
    if n == 0:
        return {"n_windows": 0, "mean_return_pct": 0.0, "median_return_pct": 0.0,
                "pct_positive": 0.0, "worst_return_pct": 0.0, "worst_mdd_pct": 0.0,
                "std_return_pct": 0.0}
    rets = [float(w.get("return_pct") or 0.0) for w in windows]
    mdds = [float(w.get("mdd_pct") or 0.0) for w in windows]
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    srt = sorted(rets)
    median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    return {"n_windows": n,
            "mean_return_pct": round(mean, 2),
            "median_return_pct": round(median, 2),
            "pct_positive": round(sum(1 for r in rets if r > 0) / n, 3),
            "worst_return_pct": round(min(rets), 2),
            "worst_mdd_pct": round(min(mdds), 2),
            "std_return_pct": round(math.sqrt(var), 2)}


def walk_forward(params: dict, prices: Dict[str, List[dict]], *,
                 test_days: int = 20, warmup_days: int = 20, step_days: int = None) -> Dict:
    """연속(비중첩 기본) 윈도우로 run_backtest 를 반복. 각 윈도우는 warmup_days(지표 워밍업)+
    test_days 길이. 반환 {windows:[{start,end,return_pct,mdd_pct,trades,win_rate_pct}], aggregate}."""
    step = int(step_days or test_days)
    all_dates = sorted({d["date"] for rows in prices.values() for d in rows})
    win_len = int(warmup_days) + int(test_days)
    windows: List[Dict] = []
    i = 0
    while i + win_len <= len(all_dates):
        wdates = set(all_dates[i:i + win_len])
        sliced = {c: [r for r in rows if r["date"] in wdates] for c, rows in prices.items()}
        sliced = {c: rows for c, rows in sliced.items() if len(rows) > warmup_days}
        if sliced:
            m = run_backtest(params, sliced, lookback=int(warmup_days))
            windows.append({"start": all_dates[i], "end": all_dates[i + win_len - 1],
                            "return_pct": m["total_return_pct"], "mdd_pct": m["max_drawdown_pct"],
                            "trades": m["trades"], "win_rate_pct": m["win_rate_pct"]})
        i += step
    return {"windows": windows, "aggregate": _aggregate(windows)}
