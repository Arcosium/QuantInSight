"""잔차 lead-lag 맵 — 종목 간 선행-후행(시차상관) 신호. stat-arb 확신 게이트용.

원 ArcTrade 그랜저 엔진의 경량 재현: 그랜저 F검정 대신 **잔차 시차상관 + 홀드아웃 재현검증**
(원 시스템도 최종 순위는 시차상관으로 매겼음 — p값이 포화됐기 때문). 시장중립 잔차 위에서
계산해 '진짜' 개별 선행-후행만 남긴다(같이 움직이는 시장/섹터 공통이동은 to_residuals 가 제거).

쓰임: build_map(일1회) → {follower: [{leader,lag,corr}]} 캐시. 실시간은 confirm() 로 dict 조회 —
과매도 후보 B 의 선행주가 방금 (예측 방향으로) 움직였으면 진입 확신을 준다. 지연 0.

fail-soft: 데이터 부족·실패 시 빈 맵 → 게이트 무효(stat-arb 단독으로 계속).
"""
import json
import logging

import numpy as np

import config
from core import analytics, crawler
from utils import market_time as mt

log = logging.getLogger("lag.leadlag")
CACHE = config.LEADLAG_MAP_CACHE
_mem = {"mtime": -1.0, "map": {}}


def _lagged_corr(R, k):
    """R(T×N) 잔차 → 시차 k 상관행렬 N×N, [leader, follower] = corr(leader_t, follower_{t+k})."""
    X, Y = R[:-k], R[k:]
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return (Xz.T @ Yz) / max(1, X.shape[0])


def build_map(conn, codes, upto_day=None):
    """lead-lag 맵 빌드. upto_day(YYYYMMDD) 주면 그 날 '이전' 데이터만 사용(백테스트 무-lookahead)."""
    since = analytics._lookback_since(config.LEADLAG_TRAIN_DAYS + 3)
    days = analytics.available_days(conn, since=since)
    if upto_day:
        days = [d for d in days if d < upto_day]
    days = days[-(config.LEADLAG_TRAIN_DAYS + 1):]
    if len(days) < 2:
        return {}
    train_days, hold_days = days[:-1], days[-1:]
    markets, sectors = crawler.load_markets(), crawler.load_sectors()
    p_tr = analytics.load_close_matrix(conn, codes, days=train_days)
    p_ho = analytics.load_close_matrix(conn, codes, days=hold_days)
    if p_tr.empty or p_ho.empty:
        return {}
    common = [c for c in p_tr.columns if c in p_ho.columns]
    if len(common) < 20:
        return {}
    R_tr, c_tr = analytics.to_residuals(p_tr[common], markets, sectors)
    R_ho, _ = analytics.to_residuals(p_ho[common], markets, sectors)
    if R_tr.shape[0] < config.STATARB_MIN_OBS or R_ho.shape[0] < 30:
        return {}
    N = len(c_tr)
    mn = float(config.LEADLAG_MIN_CORR)
    best = {}
    for k in config.LEADLAG_LAGS:
        if R_tr.shape[0] <= k or R_ho.shape[0] <= k:
            continue
        Ctr, Cho = _lagged_corr(R_tr, k), _lagged_corr(R_ho, k)
        mask = (np.abs(Ctr) >= mn) & (np.abs(Cho) >= mn) & (np.sign(Ctr) == np.sign(Cho))
        np.fill_diagonal(mask, False)
        li, fj = np.where(mask)                     # 통과 (leader, follower) 쌍
        for i, j in zip(li.tolist(), fj.tolist()):
            best.setdefault(c_tr[j], []).append((c_tr[i], k, float(Cho[i, j])))
    lead_map = {}
    for folw, lst in best.items():
        lst.sort(key=lambda t: -abs(t[2]))          # 홀드아웃 재현 |상관| 큰 순
        lead_map[folw] = [{"leader": l, "lag": k, "corr": round(c, 4)}
                          for l, k, c in lst[:config.LEADLAG_MAX_LEADERS]]
    if upto_day is None:                            # 라이브 빌드만 캐시(백테스트는 메모리)
        try:
            CACHE.write_text(json.dumps({"built": mt.now_kst().isoformat(timespec="seconds"),
                                         "train": train_days, "hold": hold_days,
                                         "n_followers": len(lead_map), "map": lead_map},
                                        ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    log.info("lead-lag 맵: 학습%s 홀드%s → 후행 %d종목·페어 %d",
             train_days[0] + "~" + train_days[-1], ",".join(hold_days),
             len(lead_map), sum(len(v) for v in lead_map.values()))
    return lead_map


def load_map():
    """캐시된 lead-lag 맵(파일 mtime 캐시)."""
    if not config.LEADLAG_ENABLED:
        return {}
    try:
        mtime = CACHE.stat().st_mtime
    except OSError:
        return {}
    if mtime != _mem["mtime"]:
        try:
            _mem["map"] = json.loads(CACHE.read_text(encoding="utf-8")).get("map", {})
            _mem["mtime"] = mtime
        except (OSError, ValueError):
            _mem["map"] = {}
    return _mem["map"]


def confirm(lead_map, recent_resid):
    """후행별 예측 이동 = Σ corr × 선행주 최근 잔차수익률. 양수·크면 '선행주가 상승 예측'.
    recent_resid: {code: 최근(마지막 분) 잔차수익률}. 반환 {follower: 예측값}."""
    out = {}
    if not lead_map:
        return out
    for folw, leaders in lead_map.items():
        v, n = 0.0, 0
        for L in leaders:
            r = recent_resid.get(L["leader"])
            if r is not None:
                v += L["corr"] * r
                n += 1
        if n:
            out[folw] = v
    return out


def cache_age_hours():
    import time
    try:
        return (time.time() - CACHE.stat().st_mtime) / 3600.0
    except OSError:
        return 1e9


if __name__ == "__main__":                          # 수동 빌드: python3.12 -m core.leadlag
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    conn = crawler.open_db()
    codes = [c for c, _ in crawler.load_universe()]
    m = build_map(conn, codes)
    print(f"lead-lag 맵: 후행 {len(m)}종목, 페어 {sum(len(v) for v in m.values())}")
    for f, ls in list(m.items())[:5]:
        print(" ", f, "←", [(x['leader'], x['lag'], x['corr']) for x in ls])
