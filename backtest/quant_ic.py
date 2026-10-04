"""퀀트점수 지표의 예측력(IC) 측정 — P1 (사장 지시 2026-08-02).

왜 필요한가:
  backtest/engine.py 는 '고정 SMA 프록시 + 리스크 규칙'만 평가한다(스스로 밝힌 정직성 경계 —
  종목 선정 능력은 평가하지 않는다). 그래서 **QIW_RSI=5 / QIW_MOM=12 …** 아홉 개 가중치가
  실제로 예측력이 있는지 지금껏 아무도 측정한 적이 없고, 운용지원실장의 주간 튜닝은 근거 없이
  숫자를 흔드는 일이었다. 이 모듈이 그 빈칸을 메운다.

무엇을 재는가:
  종목 선정에 실제로 쓰이는 결정론 신호(tools.quant_score.indicator_signals)를 과거 각 시점에
  전 종목 크로스섹셔널로 계산해, h영업일 후 수익률과의 **스피어만 순위상관(IC)** 을 낸다.
  축(지표)별 IC 와 합성점수 IC 를 함께 내므로 "어느 축이 밥값을 하는가"가 바로 보인다.
  IR = mean(IC)/std(IC) — 부호가 시기마다 뒤집히면(불안정) IR 이 낮게 나온다.

정직성 경계 (재현 불가 → 정직히 제외, 무음 결손 금지):
  • flow(외인·기관 수급) — 과거 시점의 investor CSV 스냅샷이 없어 재현 불가.
    ⚠️ compute_quant_indicators 는 investor=None 이면 **현재** CSV 를 읽는다(룩어헤드!) —
    반드시 investor=[] 를 넘겨 축 자체를 결손 처리한다.
  • leadlag(30분 그랜저) — 분봉 재현 비용 과다.
  • news·macro 차원 — 과거 감성·매크로 재현 불가. 여기선 QUANT 차원만 평가한다.
  룩어헤드 방지: t 시점 지표는 df[:t+1] 만 보고, 수익률은 t → t+h 종가로 계산한다.

CLI (넓은 표본은 서버 밖에서):
  python3 -m backtest.quant_ic --names 400 --dates 60
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("QUANT_IC")

# 재현 불가로 IC 측정에서 제외하는 축 (위 '정직성 경계' 참조)
EXCLUDED_AXES = ("flow", "leadlag")


def _kr_codes(limit: Optional[int] = None) -> List[str]:
    """data/daily_<6자리>.csv 의 KR 종목 코드. 크로스섹셔널 IC 는 거래일 달력이 같아야 해서
    KR 만 쓴다(US 는 codes= 로 명시 주입)."""
    from tools.market_data import DATA_DIR
    codes = sorted(p.stem[len("daily_"):] for p in DATA_DIR.glob("daily_*.csv"))
    codes = [c for c in codes if c.isdigit() and len(c) == 6]
    return codes[:limit] if limit else codes


def _frames(codes: List[str], min_rows: int) -> Dict[str, Any]:
    from tools.market_data import load_daily_csv
    out = {}
    for c in codes:
        df = load_daily_csv(c)
        if df is not None and len(df) >= min_rows:
            out[c] = df.reset_index(drop=True)
    return out


def _ic_stats(ics: List[float]) -> Dict[str, Any]:
    """[날짜별 IC] → 평균·표준편차·IR·양(+)비율. IR 이 낮으면 '평균은 좋은데 시기마다 뒤집힘'."""
    n = len(ics)
    if n == 0:
        return {"ic_mean": None, "ic_std": None, "ir": None, "hit_rate": None, "n_dates": 0}
    mean = sum(ics) / n
    std = (sum((x - mean) ** 2 for x in ics) / n) ** 0.5
    return {"ic_mean": round(mean, 4),
            "ic_std": round(std, 4),
            "ir": (round(mean / std, 3) if std > 1e-9 else None),
            "hit_rate": round(sum(1 for x in ics if x > 0) / n, 3),
            "n_dates": n}


def run_ic(weights: Optional[Dict[str, float]] = None, *, uid=None,
           codes: Optional[List[str]] = None, horizons=(5, 20),
           step: int = 5, dates: int = 16, max_names: int = 120,
           min_names: int = 20) -> Dict[str, Any]:
    """크로스섹셔널 IC 측정. 반환 {available, composite, by_axis, ...}.

    dates × max_names 가 계산량을 지배한다(주간 리뷰 인라인 실행이라 기본값은 보수적).
    """
    try:
        import runtime
        from tools.market_data import compute_quant_indicators
        from tools.quant_score import indicator_signals, compute_indicator_score
        from tools.agent_scorecard import information_coefficient
    except Exception as e:                      # 의존 모듈 부재(테스트 격리 등) — 무음 금지
        return {"available": False, "reason": f"import 실패: {e}"}

    w = dict(weights or runtime.quant_weights(uid))
    max_h = max(horizons)
    # 워밍업 63일(mom_3m) — 252일(high52)은 못 채워도 그 축만 결손 처리되므로 하한으로 쓰지 않는다.
    warmup = 63
    frames = _frames(codes or _kr_codes(max_names), min_rows=warmup + max_h + 5)
    if len(frames) < min_names:
        return {"available": False, "reason": f"종목 부족({len(frames)}<{min_names})"}

    pos = {c: {str(d)[:10]: i for i, d in enumerate(df["date"])} for c, df in frames.items()}
    closes = {c: [float(x) for x in df["close"]] for c, df in frames.items()}

    # 평가 날짜: min_names 종목 이상이 거래한 날 중, 마지막 max_h 일을 뺀 뒤 step 간격 샘플.
    cal: Dict[str, int] = {}
    for m in pos.values():
        for d in m:
            cal[d] = cal.get(d, 0) + 1
    all_dates = sorted(d for d, n in cal.items() if n >= min_names)
    usable = all_dates[warmup:len(all_dates) - max_h]
    eval_dates = usable[::-1][::step][:dates][::-1]
    if not eval_dates:
        return {"available": False, "reason": "평가 가능 날짜 없음(히스토리 부족)"}

    # 날짜별 IC 누적: 합성점수 + 축별
    comp: Dict[int, List[float]] = {h: [] for h in horizons}
    axis: Dict[str, Dict[int, List[float]]] = {}
    names_per_date: List[int] = []

    for d in eval_dates:
        rows = []                                # [(code, score, {axis: sig}, {h: fwd_ret})]
        for c, df in frames.items():
            i = pos[c].get(d)
            if i is None or i < warmup or i + max_h >= len(df):
                continue
            px = closes[c][i]
            if px <= 0:
                continue
            # ⚠️ investor=[] — None 이면 현재 시점 CSV 를 읽어 룩어헤드가 된다.
            ind = compute_quant_indicators(c, daily=df.iloc[:i + 1], investor=[])
            if not ind:
                continue
            sigs = {k: v for k, v in indicator_signals(ind).items() if k not in EXCLUDED_AXES}
            if not sigs:
                continue
            score, _ = compute_indicator_score(ind, w)
            rows.append((score, sigs, {h: closes[c][i + h] / px - 1.0 for h in horizons}))
        if len(rows) < min_names:
            continue
        names_per_date.append(len(rows))
        for h in horizons:
            ic = information_coefficient([(s, r[h]) for s, _, r in rows])
            if ic is not None:
                comp[h].append(ic)
        for ax in {k for _, sg, _ in rows for k in sg}:
            pairs_by_h = {h: [(sg[ax], r[h]) for _, sg, r in rows if ax in sg] for h in horizons}
            for h in horizons:
                if len(pairs_by_h[h]) >= min_names:
                    ic = information_coefficient(pairs_by_h[h])
                    if ic is not None:
                        axis.setdefault(ax, {}).setdefault(h, []).append(ic)

    if not names_per_date:
        return {"available": False, "reason": "표본 부족(날짜별 종목 수 미달)"}

    by_axis = {ax: {f"h{h}": _ic_stats(ics) for h, ics in per_h.items()}
               for ax, per_h in axis.items()}
    out = {"available": True,
           "n_dates": len(names_per_date),
           "names_per_date": round(sum(names_per_date) / len(names_per_date), 1),
           "period": f"{eval_dates[0]} ~ {eval_dates[-1]}",
           "weights_used": {k: v for k, v in w.items() if k not in EXCLUDED_AXES},
           "composite": {f"h{h}": _ic_stats(ics) for h, ics in comp.items()},
           "by_axis": by_axis,
           "excluded_axes": list(EXCLUDED_AXES),
           "note": "IC=스피어만 순위상관(신호 vs h영업일 후 수익률). "
                   "flow·leadlag 는 과거 재현 불가로 제외 — 이 축들의 가중치는 여기서 검증되지 않는다."}
    out["sign_conflicts"] = sign_conflicts(out)
    return out


def sign_conflicts(result: Dict[str, Any], *, horizon: str = "h20",
                   min_dates: int = 8, min_abs_ic: float = 0.01) -> List[str]:
    """가중치 부호와 실측 IC 부호가 반대인 축 — 운용지원실장이 바로 손댈 지점.

    예: QIW_MOM=+12 인데 mom 의 IC 가 −0.05 면 '모멘텀 가점'이 실제로는 역효과였다는 뜻이다.
    표본이 적거나(min_dates 미만) IC 가 잡음 수준(min_abs_ic 미만)이면 판단을 보류한다."""
    out = []
    for ax, per_h in (result.get("by_axis") or {}).items():
        st = (per_h or {}).get(horizon) or {}
        ic, nd = st.get("ic_mean"), st.get("n_dates") or 0
        w = (result.get("weights_used") or {}).get(ax)
        if ic is None or w is None or nd < min_dates or abs(ic) < min_abs_ic or float(w) == 0:
            continue
        if (float(w) > 0) != (ic > 0):
            out.append(f"{ax}: 가중치 {w:+g} vs 실측 IC {ic:+.3f} ({horizon}, {nd}일) — 부호 불일치")
    return out


def summary_lines(result: Dict[str, Any], *, top: int = 4) -> List[str]:
    """주간 보고 메시지용 요약 몇 줄."""
    if not result.get("available"):
        return [f"• 퀀트점수 IC 측정: 불가({result.get('reason', '사유 미상')})"]
    lines = []
    for hk, st in (result.get("composite") or {}).items():
        if st.get("ic_mean") is not None:
            lines.append(f"• 합성 퀀트점수 IC({hk}): {st['ic_mean']:+.3f} · IR {st.get('ir')} · "
                         f"양(+)날 {st.get('hit_rate', 0) * 100:.0f}% ({st.get('n_dates')}일)")
    ranked = sorted(((ax, (d.get("h20") or {}).get("ic_mean")) for ax, d in (result.get("by_axis") or {}).items()),
                    key=lambda kv: -(kv[1] if kv[1] is not None else 0))
    ranked = [(a, v) for a, v in ranked if v is not None]
    if ranked:
        lines.append("• 축별 IC(h20) 상위: " + ", ".join(f"{a} {v:+.3f}" for a, v in ranked[:top]))
        lines.append("• 축별 IC(h20) 하위: " + ", ".join(f"{a} {v:+.3f}" for a, v in ranked[-top:]))
    for c in (result.get("sign_conflicts") or []):
        lines.append(f"  ⚠️ {c}")
    return lines


if __name__ == "__main__":                      # 넓은 표본 오프라인 실행
    import argparse, json
    ap = argparse.ArgumentParser(description="퀀트점수 지표 IC 측정")
    ap.add_argument("--names", type=int, default=200)
    ap.add_argument("--dates", type=int, default=40)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    r = run_ic(max_names=args.names, dates=args.dates, step=args.step)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("\n".join(summary_lines(r)))
