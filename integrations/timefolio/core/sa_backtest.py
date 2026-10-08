"""stat-arb 워크포워드 백테스트 — bars.db 로 라이브 전략을 재현해 엣지·파라미터를 측정한다.

핵심: **라이브와 완전히 같은 로직**을 쓴다(analytics.evaluate_scores + core/strategy 의 청산규칙).
매 결정분마다 그 시점까지의 2세션 창으로 s-score 스냅샷을 만들어 진입/청산을 시뮬한다
(미래 데이터 안 씀 = look-ahead 없음). 라이브가 매 분 하는 일을 과거에 그대로 돌리는 것.

사용:
  python3.12 -m core.sa_backtest                 # 최근 전량, 5분 간격, 전 필터 on
  python3.12 -m core.sa_backtest --days 5 --step 3
  python3.12 -m core.sa_backtest --ablation       # 필터별 on/off 비교
"""
import argparse
import logging

import numpy as np

import config
from core import analytics, crawler, dart_risk

log = logging.getLogger("lag.backtest")
_SLIP = config.PAPER_SLIPPAGE_BPS / 10_000.0


def _metrics(trades, cost_pct):
    """청산 거래 리스트 → 지표 dict."""
    if not trades:
        return {"n": 0}
    rets = np.array([t["ret"] for t in trades])
    net = rets - cost_pct / 100.0                        # 왕복 마찰 차감
    holds = np.array([t["held"] for t in trades])
    from collections import Counter
    reasons = Counter(t["reason"] for t in trades)
    # 거래당 평균은 표준오차 없이 못 읽는다. |t|<2 면 방향조차 단정하면 안 된다.
    se = float(rets.std() / np.sqrt(len(rets))) if len(rets) > 1 else float("nan")
    return {"n": len(trades),
            "gross_sd_bps": float(rets.std() * 1e4),
            "gross_t": float(rets.mean() / se) if se and se > 0 else float("nan"),
            "gross_sum_pct": float(rets.sum() * 100), "gross_avg_pct": float(rets.mean() * 100),
            "net_sum_pct": float(net.sum() * 100), "net_avg_pct": float(net.mean() * 100),
            "win_gross_pct": float((rets > 0).mean() * 100),
            "win_net_pct": float((net > 0).mean() * 100),
            "avg_hold_min": float(holds.mean()),
            "reasons": dict(reasons)}


def run(conn, codes, ndays=None, step=5, gates=None, carry=False, momentum=False, fill_lag=1):
    """워크포워드 시뮬(절대-행). gates: {'regime','dart','sector','knife','eigen'} 중 False 면 그 필터 끔.

    fill_lag: 신호를 만든 봉으로부터 몇 봉 뒤에 체결할지. **1 이 정상이다.**
    0 = 구 동작(신호 봉의 종가로 즉시 체결) = 1봉 lookahead. 비교용으로만 남긴다.
    carry=True 면 포지션을 다음 날로 넘긴다. momentum=True 면 **역발상**: 잔차가 추세면
    저평가(반전) 대신 **고평가(s>+진입, 잔차 강세)를 매수**하고 s가 식으면(s<+청산) 청산 = 잔차 모멘텀."""
    gates = gates or {}
    days = analytics.available_days(conn)
    if ndays:
        days = days[-int(ndays):]
    if len(days) < 2:
        return {"n": 0, "error": "영업일 부족"}
    full = analytics.load_close_matrix(conn, codes, days=days)
    if full.empty:
        return {"n": 0, "error": "데이터 없음"}
    common = list(full.columns)
    tv = analytics.avg_daily_turnover(conn, common, days)
    liquid = [c for c in common if tv.get(c, 0.0) >= config.MIN_DAILY_TURNOVER_KRW]
    if len(liquid) >= 10:
        common = liquid
    full = full[common]
    markets = crawler.load_markets()
    sectors = analytics.sector_map() if gates.get("sector", True) else None
    risky = dart_risk.risky_set() if gates.get("dart", True) else set()

    # ablation: config 임시 오버라이드(레짐/낙하칼 끄기)
    saved = {}
    if not gates.get("eigen", True):
        saved["USE_EIGEN_FACTORS"], config.USE_EIGEN_FACTORS = config.USE_EIGEN_FACTORS, 0
    if not gates.get("regime", True):
        saved["STATARB_MAX_TREND_ER"], config.STATARB_MAX_TREND_ER = config.STATARB_MAX_TREND_ER, 1e9
        saved["STATARB_MARKET_FALL"], config.STATARB_MARKET_FALL = config.STATARB_MARKET_FALL, 1e9
    if not gates.get("knife", True):
        saved["STATARB_MAX_ABS_DROP"], config.STATARB_MAX_ABS_DROP = config.STATARB_MAX_ABS_DROP, 1e9
    if not gates.get("sector", True):
        saved["USE_SECTOR_FACTOR"], config.USE_SECTOR_FACTOR = config.USE_SECTOR_FACTOR, 0

    S_EXIT, S_STOP = config.STATARB_S_EXIT, config.STATARB_S_STOP
    SL, TP, MAXPOS = config.STOP_LOSS, config.TAKE_PROFIT, config.MAX_POSITIONS
    import math
    win_rows = int(config.STATARB_WINDOW_MIN + math.ceil(config.STATARB_WINDOW_MIN / 390.0) + 5)
    idx = list(full.index)
    npx = full.to_numpy(dtype=np.float64)            # 빠른 가격 조회
    col = {c: j for j, c in enumerate(full.columns)}
    trades, positions = [], {}
    lead_map, cur_day = {}, None
    use_ll = config.LEADLAG_ENABLED and gates.get("leadlag", True)
    try:
        for t in range(win_rows, len(idx), max(1, step)):
            win = full.iloc[max(0, t - win_rows):t + 1]
            if win.shape[0] < config.STATARB_MIN_OBS:
                continue
            day = idx[t][:8]
            if use_ll and day != cur_day:           # 일 1회 맵 재빌드(그날 이전 데이터만 = 무-lookahead)
                cur_day = day
                from core import leadlag
                lead_map = leadlag.build_map(conn, codes, upto_day=day)
            snap = analytics.evaluate_scores(win, markets, sectors, risky,
                                             lead_map=lead_map, updated=idx[t])
            if not snap:
                continue
            sc = snap["scores"]
            # 청산(우선). held = 행거리(세션분, 밤 제외).
            # 체결은 **다음 봉**이다. t 의 종가로 신호를 만들고 같은 t 종가에 체결하면
            # 1봉 lookahead 다 — 크립토 판에서 이걸 고치자 총이익이 3.46→2.60bp 로 떨어졌다
            # (엣지의 4분의 1). KRX 판에도 같은 버그가 있었다(2026-08-07).
            tf = t + fill_lag
            if tf >= len(idx):
                break
            for code in list(positions):
                p = positions[code]
                cur = npx[tf, col[code]] if code in col else 0.0
                if not (cur > 0):
                    continue
                s = sc.get(code, {}).get("s")
                pnl = cur / p["entry"] - 1.0
                held = t - p["t"]
                reason = None
                if momentum:                         # 모멘텀: s>+진입에 사서 s<+청산(강세 소멸)에 판다
                    if s is not None and s < config.STATARB_S_EXIT:
                        reason = "FADE"
                elif s is not None and s > -S_EXIT:  # 반전(기본): s>-청산이면 회귀완료
                    reason = "REVERT"
                elif s is not None and s < -S_STOP:
                    reason = "SSTOP"
                if reason is None:
                    if pnl <= SL:
                        reason = "SL"
                    elif pnl >= TP:
                        reason = "TP"
                    elif p["hl"] > 0 and held >= p["hl"]:
                        reason = "TIMEOUT"
                    elif not carry and idx[t][:8] != idx[p["t"]][:8]:
                        reason = "EOD"                # carry off: 익일 첫 결정에 청산
                if reason:
                    trades.append({"ret": float(cur) * (1 - _SLIP) / p["entry"] - 1.0,
                                   "held": held, "reason": reason, "day": idx[t][:8]})
                    del positions[code]
            # 진입: 모멘텀이면 고평가(shorts=s>+진입) 매수, 아니면 저평가(candidates=s<-진입) 매수
            room = MAXPOS - len(positions)
            for c in (snap["shorts"] if momentum else snap["candidates"])[:max(0, room)]:
                if c["code"] in positions or c["code"] not in col:
                    continue
                entry = npx[tf, col[c["code"]]]             # 진입도 같은 시점 (lookahead 차단)
                if entry > 0:
                    positions[c["code"]] = {"entry": entry * (1 + _SLIP),
                                            "t": t, "hl": c["half_life"], "s": c["s"]}
    finally:
        for k, v in saved.items():           # config 원복
            setattr(config, k, v)

    m = _metrics(trades, float(config.AUTOFOLIO_ROUND_TRIP_COST_PCT or 0.0))
    m["days"] = f"{days[1]}~{days[-1]}"
    m["universe"] = len(common)
    m["_trades"] = trades           # 레짐 버킷 분석용(day 태그 포함)
    return m


def _fmt(m):
    if m.get("n", 0) == 0:
        return f"  거래 0건 ({m.get('error','후보 없음')})"
    return (f"  거래 {m['n']}건 · 승률(gross) {m['win_gross_pct']:.0f}% · 승률(net) {m['win_net_pct']:.0f}%"
            f" · gross t={m.get('gross_t', float('nan')):+.2f} (sd {m.get('gross_sd_bps', 0):.0f}bp)\n"
            f"  Gross 합 {m['gross_sum_pct']:+.2f}% (평균 {m['gross_avg_pct']:+.3f}%) · "
            f"Net 합 {m['net_sum_pct']:+.2f}% (평균 {m['net_avg_pct']:+.3f}%)\n"
            f"  평균보유 {m['avg_hold_min']:.1f}분 · 청산사유 {m['reasons']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    conn = crawler.open_db()
    codes = [c for c, _ in crawler.load_universe()]
    base = run(conn, codes, a.days, a.step)
    print(f"\n=== stat-arb 백테스트 ({base.get('days','?')} · 유니버스 {base.get('universe','?')} · {a.step}분 간격) ===")
    print("[전 필터 ON]")
    print(_fmt(base))
    if a.ablation:
        for name, g in [("고유포트폴리오 OFF(구 KOSPI/KOSDAQ 2팩터)", {"eigen": False}),
                        ("레짐 OFF", {"regime": False}), ("DART OFF", {"dart": False}),
                        ("섹터팩터 OFF", {"sector": False}), ("낙하칼 OFF", {"knife": False}),
                        ("전 필터 OFF", {"regime": False, "dart": False, "sector": False, "knife": False})]:
            print(f"\n[{name}]")
            print(_fmt(run(conn, codes, a.days, a.step, gates=g)))
