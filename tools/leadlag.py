"""그랜저 선행-후행(lead-lag) 신호 — 30분 지평 (사장 지시 2026-07-21).

배경: 자체 분봉 크롤러(market_bars/)가 KOSPI200+KOSDAQ150 350종목의 **분봉**을 data/bars.db 에
매분 수집한다. 분(minute) 단위 선행-후행은 QuantInSight 사이클(20분+)엔 무용하므로 → 여기서는
분봉을 **30분 버킷**으로 리샘플해 30분 지평의 선행-후행 관계를 재구성한다(시차 ≥ 30분).
"선행주 A가 최근 30분에 (시장대비) 오르면, 그 후행주 B가 다음 30분에 따라 오른다"는 신호를
후행주 B의 **계량 팩터**(s∈[-1,+1])로 제공한다.

원칙:
  · 읽기 전용 — bars.db 를 읽기만 한다(수집은 market_bars 크롤러가 담당).
  · 시장중립 — 각 30분 버킷의 횡단면 평균수익률을 빼(공통 시장요인 제거) '진짜' 상대 선행-후행만 본다.
  · fail-soft — 어떤 실패도 0/[] 를 돌려주고 사이클을 절대 막지 않는다.
  · 캐시 — 구조적 선행-후행 맵은 무겁다(349² 상관). 1시간 캐시. 최근수익률은 5분 캐시.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("LEADLAG")

_MAP_CACHE: Dict[str, Any] = {"ts": 0.0, "map": {}}       # {follower: [(leader, corr), ...]}
_RET_CACHE: Dict[str, Any] = {"ts": 0.0, "ret": {}}       # {code: 최근 30분 잔차수익률}


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _bars_db() -> str:
    import os
    _default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bars.db")
    return _cfg("LEADLAG_BARS_DB", _default)


def _bucket(ts: str) -> str:
    """분봉 ts 'YYYYMMDDHHMM' → 30분 버킷 키 'YYYYMMDDHH'+('00'|'30')."""
    ts = str(ts)
    return ts[:10] + ("00" if int(ts[10:12]) < 30 else "30")


def _load_30m_returns(days: int, market_neutral: bool = True) -> Tuple[List[str], "Any"]:
    """최근 days 영업일의 30분 버킷 종가 → 30분 수익률 행렬 (codes, R[T×N]).
    market_neutral=True 면 각 버킷 횡단면 평균 제거(선행-후행용). False 면 원수익률(상관 분산용).
    실패/데이터부족 → ([], None)."""
    try:
        import numpy as np
    except Exception:
        return [], None
    db = _bars_db()
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        logger.debug("bars.db 열기 실패: %s", e)
        return [], None
    try:
        # 최근 days 영업일 범위: 존재하는 날짜(YYYYMMDD) 상위 days개
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT substr(ts,1,8) d FROM bars ORDER BY d DESC LIMIT ?", (int(days),))]
        if not dates:
            return [], None
        lo = min(dates)
        rows = conn.execute(
            "SELECT code, ts, close FROM bars WHERE substr(ts,1,8) >= ? ", (lo,)).fetchall()
    except Exception as e:  # noqa: BLE001
        logger.debug("bars.db 조회 실패: %s", e)
        return [], None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not rows:
        return [], None
    # 30분 버킷별 마지막 종가
    buck: Dict[str, Dict[str, float]] = {}   # {code: {bucket: last_close}}
    for code, ts, close in rows:
        b = _bucket(ts)
        d = buck.setdefault(str(code), {})
        d[b] = float(close)   # 같은 버킷 후행 ts 가 마지막 종가로 덮어씀(정렬 무관 최종값 아님 → 아래서 정렬)
    # 정렬 보장: 버킷별 '최대 ts'의 종가를 쓰려면 재수집 필요 → 간단화: ts 정렬 후 마지막
    buck = {}
    rows.sort(key=lambda r: (str(r[0]), str(r[1])))
    for code, ts, close in rows:
        buck.setdefault(str(code), {})[_bucket(ts)] = float(close)
    all_buckets = sorted({b for d in buck.values() for b in d})
    if len(all_buckets) < 6:
        return [], None
    codes = sorted(buck.keys())
    bidx = {b: i for i, b in enumerate(all_buckets)}
    import numpy as np
    P = np.full((len(all_buckets), len(codes)), np.nan)
    for j, c in enumerate(codes):
        for b, cl in buck[c].items():
            P[bidx[b], j] = cl
    # 30분 수익률 (버킷 간). 결측은 이전 버킷 대비만 계산 → NaN 유지.
    R = np.full((P.shape[0] - 1, P.shape[1]), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (P[1:] / P[:-1]) - 1.0
    # 시장중립: 각 버킷(행)의 횡단면 평균 제거(공통 시장요인 제거) — 선행-후행용만
    if market_neutral:
        mkt = np.nanmean(R, axis=1, keepdims=True)
        R = R - mkt
    return codes, R


def _build_map() -> Dict[str, List[Tuple[str, float]]]:
    """구조적 30분 선행-후행 맵 {follower: [(leader, corr), ...]} (시차 1버킷=30분).
    corr(A_{t-1}, B_t) 가 임계 이상이고 방향성(A→B > B→A)이면 A 를 B 의 선행주로 채택."""
    try:
        import numpy as np
    except Exception:
        return {}
    days = int(_cfg("LEADLAG_LOOKBACK_DAYS", 10) or 10)
    min_conf = float(_cfg("LEADLAG_MIN_CONF", 0.30) or 0.30)
    top_k = int(_cfg("LEADLAG_TOP_LEADERS", 3) or 3)
    codes, R = _load_30m_returns(days)
    if not codes or R is None or R.shape[0] < 8:
        return {}
    # 표준화(열별) — NaN 은 0 으로(결측 버킷은 신호 없음)
    Z = np.where(np.isfinite(R), R, 0.0)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd
    T = Z.shape[0]
    lead = Z[:-1]     # A_{t-1}
    folw = Z[1:]      # B_t
    # C[a,b] = corr(A_{t-1}, B_t) = A 가 B 를 선행
    C = (lead.T @ folw) / max(1, (T - 1))
    np.fill_diagonal(C, 0.0)
    out: Dict[str, List[Tuple[str, float]]] = {}
    Ct = C.T
    N = len(codes)
    for b in range(N):
        leaders = []
        col = C[:, b]      # 모든 A 에 대해 A→b
        for a in range(N):
            c_ab = float(col[a])
            if c_ab < min_conf:
                continue
            if c_ab <= float(C[b, a]):   # 방향성: A→B 가 B→A 보다 강해야 진짜 선행
                continue
            leaders.append((codes[a], round(c_ab, 3)))
        if leaders:
            leaders.sort(key=lambda x: -x[1])
            out[codes[b]] = leaders[:top_k]
    logger.info("선행-후행 맵 구축: 후행주 %d개 (30분 지평, %d영업일)", len(out), days)
    return out


def _get_map() -> Dict[str, List[Tuple[str, float]]]:
    ttl = float(_cfg("LEADLAG_MAP_TTL_SEC", 3600) or 3600)
    now = time.time()
    if (now - _MAP_CACHE["ts"]) > ttl or not _MAP_CACHE["map"]:
        try:
            m = _build_map()
            if m:
                _MAP_CACHE["map"] = m
                _MAP_CACHE["ts"] = now
        except Exception as e:  # noqa: BLE001
            logger.warning("선행-후행 맵 구축 실패(캐시 유지): %s", e)
    return _MAP_CACHE["map"] or {}


def _recent_residual_returns() -> Dict[str, float]:
    """최근 lookback 분(≥30) 동안 각 종목의 **시장중립 30분 수익률**(선행주 최근 이동 판별용)."""
    ttl = 300.0
    now = time.time()
    if (now - _RET_CACHE["ts"]) <= ttl and _RET_CACHE["ret"]:
        return _RET_CACHE["ret"]
    try:
        import numpy as np
    except Exception:
        return {}
    lookback = max(30, int(_cfg("LEADLAG_LOOKBACK_MIN", 30) or 30))
    db = _bars_db()
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    except Exception:
        return {}
    try:
        # 최근 (lookback + 40)분 분봉만
        maxts = conn.execute("SELECT MAX(ts) FROM bars").fetchone()[0]
        if not maxts:
            return {}
        # 문자열 ts 기준 하한: 대략 최근 2*lookback+60분 넉넉히
        span = lookback * 2 + 60
        rows = conn.execute(
            "SELECT code, ts, close FROM bars WHERE ts >= ? ",
            (_ts_minus(str(maxts), span),)).fetchall()
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not rows:
        return {}
    rows.sort(key=lambda r: (str(r[0]), str(r[1])))
    # 각 종목: 최근 close vs (lookback분 전 close) → 원수익률
    from collections import defaultdict
    ser: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for code, ts, close in rows:
        ser[str(code)].append((str(ts), float(close)))
    cutoff = _ts_minus(str(maxts), lookback)
    raw: Dict[str, float] = {}
    for code, lst in ser.items():
        if len(lst) < 2:
            continue
        last_ts, last_cl = lst[-1]
        # cutoff 이하(≈lookback분 전) 가장 가까운 종가
        base = None
        for ts, cl in lst:
            if ts <= cutoff:
                base = cl
            else:
                break
        if base is None:
            base = lst[0][1]
        if base and base > 0:
            raw[code] = (last_cl / base) - 1.0
    if not raw:
        return {}
    # 시장중립: 전체 평균 제거
    import numpy as np
    m = float(np.mean(list(raw.values())))
    resid = {c: (v - m) for c, v in raw.items()}
    _RET_CACHE["ret"] = resid
    _RET_CACHE["ts"] = now
    return resid


def _ts_minus(ts: str, minutes: int) -> str:
    """'YYYYMMDDHHMM' 에서 minutes 분 뺀 대략적 하한 문자열(같은 날 가정, 자정 넘으면 날짜만 -1일)."""
    try:
        from datetime import datetime, timedelta
        dt = datetime.strptime(str(ts)[:12], "%Y%m%d%H%M") - timedelta(minutes=int(minutes))
        return dt.strftime("%Y%m%d%H%M")
    except Exception:
        return "000000000000"


def leadlag_signal(code: str) -> float:
    """후행주 code 의 30분 선행-후행 신호 s∈[-1,+1]. 선행주가 최근(≥30분) 시장대비 오르면 +.
    맵/데이터 없거나 후행주 아님 → 0.0. 절대 예외 안 던짐."""
    if not bool(_cfg("ENABLE_LEADLAG_SIGNAL", True)):
        return 0.0
    try:
        code = str(code).strip().zfill(6)
        leaders = _get_map().get(code)
        if not leaders:
            return 0.0
        rets = _recent_residual_returns()
        if not rets:
            return 0.0
        scale = float(_cfg("LEADLAG_MOVE_SCALE_PCT", 1.0) or 1.0)   # 선행주 잔차수익률 이 %면 만점 기여
        num = 0.0
        den = 0.0
        for lead, conf in leaders:
            r = rets.get(str(lead).zfill(6))
            if r is None:
                continue
            move = max(-1.0, min(1.0, (r * 100.0) / max(0.1, scale)))
            num += conf * move
            den += conf
        if den <= 0:
            return 0.0
        return max(-1.0, min(1.0, num / den))
    except Exception as e:  # noqa: BLE001
        logger.debug("leadlag_signal(%s) 실패: %s", code, e)
        return 0.0


def leadlag_candidates(top_n: int = 5, min_signal: float = 0.35) -> List[Dict[str, Any]]:
    """선행주가 최근 강하게 오른 후행주 후보(신규 매수 후보용). [{code, signal, leaders}] 내림차순.
    ENABLE_LEADLAG_SIGNAL off → []. 절대 예외 안 던짐."""
    if not bool(_cfg("ENABLE_LEADLAG_SIGNAL", True)):
        return []
    try:
        m = _get_map()
        if not m:
            return []
        out = []
        for folw in m:
            s = leadlag_signal(folw)
            if s >= float(min_signal):
                out.append({"code": folw, "signal": round(s, 3),
                            "leaders": [l for l, _ in m[folw]]})
        out.sort(key=lambda x: -x["signal"])
        return out[:max(0, int(top_n))]
    except Exception as e:  # noqa: BLE001
        logger.debug("leadlag_candidates 실패: %s", e)
        return []


# ── 포트폴리오 상관 분산 (사장 지시 2026-07-21) — 원(raw) 30분 수익률 상관 ──────────
_CORR_CACHE: Dict[str, Any] = {"ts": 0.0, "codes": [], "R": None}


def _raw_returns_cached():
    ttl = float(_cfg("LEADLAG_MAP_TTL_SEC", 3600) or 3600)
    now = time.time()
    if (now - _CORR_CACHE["ts"]) > ttl or _CORR_CACHE["R"] is None:
        try:
            codes, R = _load_30m_returns(int(_cfg("LEADLAG_LOOKBACK_DAYS", 10) or 10), market_neutral=False)
            _CORR_CACHE.update({"ts": now, "codes": codes, "R": R})
        except Exception as e:  # noqa: BLE001
            logger.debug("raw returns 로드 실패: %s", e)
    return _CORR_CACHE["codes"], _CORR_CACHE["R"]


def max_correlation_with(code: str, holding_codes) -> float:
    """후보 code 와 보유종목들 사이 30분 수익률 상관의 최대값(0~1). 데이터 없으면 0. 절대 예외 안 던짐."""
    try:
        import numpy as np
        code = str(code).strip().zfill(6)
        hold = [str(h).strip().zfill(6) for h in (holding_codes or []) if str(h).strip().zfill(6) != code]
        if not hold:
            return 0.0
        codes, R = _raw_returns_cached()
        if not codes or R is None:
            return 0.0
        idx = {c: i for i, c in enumerate(codes)}
        if code not in idx:
            return 0.0
        cvec = R[:, idx[code]]
        best = 0.0
        for h in hold:
            if h not in idx:
                continue
            hv = R[:, idx[h]]
            mask = np.isfinite(cvec) & np.isfinite(hv)
            if int(mask.sum()) < 8:
                continue
            a = cvec[mask]; b = hv[mask]
            sa = float(a.std()); sb = float(b.std())
            if sa == 0 or sb == 0:
                continue
            corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
            if corr > best:
                best = corr
        return max(0.0, min(1.0, best))
    except Exception as e:  # noqa: BLE001
        logger.debug("max_correlation_with(%s) 실패: %s", code, e)
        return 0.0


def status() -> Dict[str, Any]:
    """진단용 — 맵 크기·최근수익률 표본·설정."""
    m = _get_map()
    r = _recent_residual_returns()
    return {"enabled": bool(_cfg("ENABLE_LEADLAG_SIGNAL", True)),
            "followers": len(m), "recent_return_samples": len(r),
            "lookback_min": int(_cfg("LEADLAG_LOOKBACK_MIN", 30) or 30),
            "min_conf": float(_cfg("LEADLAG_MIN_CONF", 0.30) or 0.30),
            "bars_db": _bars_db()}
