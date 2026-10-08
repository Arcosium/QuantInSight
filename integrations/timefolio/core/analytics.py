"""모듈 B — 통계적 차익거래(잔차 평균회귀) s-score 연산 엔진.

2026-07-28 리드-랙(그랜저) → stat-arb 전면 재구성. Avellaneda & Lee (2010),
"Statistical Arbitrage in the US Equities Market", Quantitative Finance 10(7).

파이프라인 (매 분):
  1) SQLite 롤링 버퍼 → 최근 영업일 정규 분봉 종가 행렬 로드, ffill
  2) 로그수익률 → **시장중립 잔차** ε_i(t): 2팩터(KOSPI·KOSDAQ 등가중평균) OLS 잔차
     (to_residuals — 리드-랙 엔진에서 그대로 재사용. 이게 논문 1단계 팩터회귀에 해당)
  3) 최근 W분 잔차를 누적 X_i(n)=Σ_{k≤n} ε_i(k)
  4) X 를 AR(1) 로 적합 → OU 파라미터:  κ=-ln(b), 반감기=ln2/κ(분), m=a/(1-b),
     σ_eq=√(Var(ξ)/(1-b²)),  **s-score** s_i=(X_i(last)-m_i)/σ_eq,i
  5) 거래 적격: 평균회귀 유의(DF t on b-1 ≤ 문턱) AND 반감기 빠름(< 창×MAX_HL_FRAC)

s 가 크게 음수 = 팩터대비 저평가(과매도) → 상승 반전 기대 → 롱 후보.
현물 롱온리라 s>+문턱(고평가) 숏 다리는 취하지 않는다.
"""
import json
import logging
import math
import time
from datetime import timedelta

import numpy as np
import pandas as pd

import config
from core import crawler, dart_risk
from utils import market_time as mt

log = logging.getLogger("lag.analytics")


# ── 1) 데이터 로드 (리드-랙 엔진에서 재사용) ─────────────────────
def available_days(conn, since=None):
    """DB 에 쌓여 있는 영업일(YYYYMMDD) 오름차순. since 로 훑는 범위를 잘라 아카이브가 커져도 싸게."""
    if since:
        rows = conn.execute(
            "SELECT DISTINCT substr(ts,1,8) FROM bars WHERE ts >= ? ORDER BY 1", (since,))
    else:
        rows = conn.execute("SELECT DISTINCT substr(ts,1,8) FROM bars ORDER BY 1")
    return [r[0] for r in rows]


def archive_stats(conn):
    """아카이브 규모(분봉은 자산·영구보존). 대시보드 표시용."""
    row = conn.execute(
        "SELECT COUNT(*), MIN(substr(ts,1,8)), MAX(substr(ts,1,8)) FROM bars").fetchone()
    rows_n, first, last = (row or (0, None, None))
    days = len(available_days(conn))
    return {"rows": int(rows_n or 0), "days": days, "first_day": first, "last_day": last}


def _lookback_since(days_needed):
    """필요 영업일 수를 담을 만큼만 과거로 거슬러 올라가는 ts 하한(주말·공휴일 여유 2배)."""
    back = max(10, int(days_needed) * 2 + 10)
    start = (mt.now_kst() - timedelta(days=back)).strftime("%Y%m%d")
    return start + "0000"


def load_close_matrix(conn, codes, days=None):
    """정규 분봉 그리드(09:01~15:19)에 정렬된 종가 행렬. 누락 분봉은 전분 종가 ffill.
    days 를 주면 그 영업일들만, 없으면 최근 BUFFER_DAYS 일. 날짜 범위를 SQL 로 잘라 PK(code,ts)를 탄다."""
    if days is None:
        days = available_days(conn, since=_lookback_since(config.BUFFER_DAYS))[-config.BUFFER_DAYS:]
    days = sorted(set(days or []))
    if not days or not codes:
        return pd.DataFrame()
    lo, hi = days[0] + "0000", days[-1] + "2359"
    rows = conn.execute(
        "SELECT code, ts, close FROM bars WHERE ts BETWEEN ? AND ? AND code IN (%s)"
        % ",".join("?" * len(codes)), [lo, hi, *codes]).fetchall()
    if not rows:
        return pd.DataFrame()
    keep = set(days)
    rows = [r for r in rows if r[1][:8] in keep]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["code", "ts", "close"])
    piv = df.pivot_table(index="ts", columns="code", values="close", aggfunc="last")
    grid = []
    for d in days:
        grid.extend(mt.session_minutes(pd.Timestamp(d).date()))
    piv = piv.reindex(grid)                     # 동시호가/그리드 밖 분봉 배제
    # 장중엔 오늘 세션의 '미래' 분(아직 도달 안 한 그리드 행)이 전부 NaN 이라, 그대로 두면 60% 유효
    # 필터가 전 종목을 떨군다 → 개장~오전 엔진 정지·스냅샷 동결(2026-07-29 실측). 마지막 데이터 행까지만.
    filled = piv.dropna(how="all").index
    if len(filled):
        piv = piv.loc[:filled.max()]
    valid = piv.notna().mean() > 0.6
    piv = piv.loc[:, valid[valid].index]
    piv = piv.ffill().bfill()
    return piv


def avg_daily_turnover(conn, codes, days):
    """학습창 각 종목 평균 일거래대금(원) → {code: 원}. ≈ AVG(close)×MAX(vol_cum). 저유동 제외용."""
    if not codes or not days:
        return {}
    days = sorted(set(days))
    lo, hi = days[0] + "0000", days[-1] + "2359"
    rows = conn.execute(
        "SELECT code, substr(ts,1,8), AVG(close)*MAX(vol_cum) FROM bars "
        "WHERE ts BETWEEN ? AND ? AND code IN (%s) GROUP BY code, substr(ts,1,8)"
        % ",".join("?" * len(codes)), [lo, hi, *codes]).fetchall()
    keep = set(days)
    agg = {}
    for code, d, amt in rows:
        if d in keep and amt:
            agg.setdefault(code, []).append(float(amt))
    return {c: sum(v) / len(v) for c, v in agg.items()}


# ── 2) 수익률 → 2팩터 시장중립 잔차 (리드-랙 엔진에서 재사용) ────
def _market_factors(R, codes, markets):
    """R(T×N) → 시장 팩터 T×2 (열0=KOSPI 등가중평균, 열1=KOSDAQ 등가중평균). 한쪽 없으면 None."""
    if not markets:
        return None
    mk = np.array([str(markets.get(c, "")).upper() for c in codes])
    kospi, kosdaq = mk == "KOSPI", mk == "KOSDAQ"
    if not kospi.any() or not kosdaq.any():
        return None
    return np.column_stack([R[:, kospi].mean(axis=1), R[:, kosdaq].mean(axis=1)])


def _normalize_tod_vol(resid, kept_ts):
    """분(HHMM) 시간대별 변동성으로 잔차를 등분산화(개장·마감 U자 완화). 얕은 슬롯(<5행) 불변, [0.25,4] 클립."""
    if resid.shape[0] == 0:
        return resid
    slots = np.array([str(t)[8:12] for t in kept_ts])
    gstd = float(resid.std()) or 1.0
    scale = np.ones(resid.shape[0])
    for s in np.unique(slots):
        m = slots == s
        if int(m.sum()) < 5:
            continue
        sig = float(resid[m].std())
        if sig > 1e-12:
            scale[m] = gstd / sig
    scale = np.clip(scale, 0.25, 4.0)
    return resid * scale[:, None]


def sector_map():
    """섹터 디민에 쓸 라벨. config.SECTOR_SOURCE 로 출처를 고른다.

    "cluster"(기본): 가격 상관으로 발견한 군(core/clusters.py). 캐시가 아직 없으면
    거래소 업종으로 폴백한다 — 첫 기동에 섹터 팩터가 통째로 꺼지는 것보다 낫다.
    "csv": universe.csv 의 거래소 업종(구 동작).
    """
    if config.SECTOR_SOURCE == "cluster":
        from core import clusters
        m = clusters.load_map()
        if m:
            return m
        log.info("클러스터 캐시 없음 — 거래소 업종으로 폴백")
    return crawler.load_sectors()


def _sector_demean(resid, codes, sectors):
    """각 바(bar)에서 종목의 '자기 섹터 동시점 평균 잔차'를 뺀다 → 섹터 공통이동 제거.
    시장 2팩터가 못 잡는 업종 로테이션(예: 반도체가 통째로 눌림)을 개별 저평가로 오인하지 않게 한다.
    멤버 2개 이상 섹터만(1개는 자기 자신이 곧 평균이라 빼면 0). Avellaneda-Lee 의 섹터 팩터 역할."""
    labels = np.array([sectors.get(c, "") for c in codes])
    out = resid.copy()
    for s in {x for x in labels if x}:
        m = labels == s
        if int(m.sum()) >= 2:
            out[:, m] -= resid[:, m].mean(axis=1, keepdims=True)
    return out


def to_residuals(prices, markets=None, sectors=None):
    """종가 행렬 → (잔차 행렬(T'×N), 코드 리스트). 일 경계 넘는 수익률 행 제거.

    팩터 단계는 두 갈래다(config.USE_EIGEN_FACTORS):
      · 1(기본, 2026-08-07~) **고유포트폴리오** 상위 k개 — crypto 판과 같은 수학.
        지수 등가중평균은 대형주에 끌려다니고 시장 구분(KOSPI/KOSDAQ)에 의존한다.
      · 0 구(舊) 2팩터(KOSPI·KOSDAQ 등가중평균). 한쪽 시장뿐이면 1팩터 폴백.
    이어서 config.NORMALIZE_TOD_VOL 이면 시간대 변동성 등분산화,
    sectors 가 있으면 섹터/클러스터 동시점 디민. (Avellaneda-Lee 팩터회귀 단계)"""
    codes = list(prices.columns)
    P = prices.to_numpy(dtype=np.float64)
    ts = list(prices.index)
    same_day = np.array([ts[i][:8] == ts[i + 1][:8] for i in range(len(ts) - 1)])
    R = np.nan_to_num(np.log(P[1:] / P[:-1])[same_day], nan=0.0, posinf=0.0, neginf=0.0)
    if R.shape[0] == 0:
        return R, codes
    kept_ts = [ts[i + 1] for i in range(len(ts) - 1) if same_day[i]]
    Rc = R - R.mean(axis=0)                      # 열별 디민 (알파 제거)

    resid = None
    if config.USE_EIGEN_FACTORS:
        from core import factors
        resid = factors.residualize(R, config.EIGEN_FACTORS)
    else:
        F = _market_factors(R, codes, markets)   # T×2 또는 None
        if F is not None:
            Fc = F - F.mean(axis=0)
            try:
                B = np.linalg.solve(Fc.T @ Fc, Fc.T @ Rc)   # 2×N 벡터화 다중회귀
                resid = Rc - Fc @ B
            except np.linalg.LinAlgError:
                resid = None
    if resid is None:
        m = R.mean(axis=1)                       # 폴백: 유니버스 등가중 평균 1팩터
        mc = m - m.mean()
        var_m = float(mc @ mc)
        resid = R if var_m <= 0 else Rc - np.outer(mc, (mc @ Rc) / var_m)

    if config.NORMALIZE_TOD_VOL:
        resid = _normalize_tod_vol(resid, kept_ts)
    if config.USE_SECTOR_FACTOR and sectors:      # 섹터 동시점 공통이동 제거(3번째 팩터)
        resid = _sector_demean(resid, codes, sectors)
    return resid, codes


# ── 3)+4) OU 적합 → s-score ──────────────────────────────────────
def ou_score(x, p=None):
    """누적 잔차 시계열 x(길이 n) → OU/s-score dict 또는 None.

    AR(1):  x_{t+1} = a + b·x_t + ξ  (OLS).  평균회귀는 0<b<1.
      κ = -ln(b) [/봉],  반감기 = ln2/κ [봉],  m = a/(1-b),  σ_eq = √(Var(ξ)/(1-b²))
      s = (x_last - m)/σ_eq
    유의성: (b-1) 의 Dickey-Fuller t 통계 (단위근 귀무 기각 = 평균회귀).
    exp_ret: 청산문턱까지 잡을 수 있는 기대 반전폭 ≈ (|s|-S_EXIT)·σ_eq (롱 후보 s<0 에만).

    p: STATARB_* 문턱을 담은 객체(기본 config). 크립토 엔진은 봉 단위·문턱이 달라
    자기 파라미터를 넘긴다 — 수학은 자산군과 무관하므로 함수는 하나만 둔다.
    시간 단위는 '봉'이다. KRX 는 1분봉이라 분과 같았을 뿐이다.
    """
    p = p or config
    n = len(x)
    if n < p.STATARB_MIN_OBS:
        return None
    x0, x1 = x[:-1], x[1:]
    x0m, x1m = float(x0.mean()), float(x1.mean())
    sxx = float(np.dot(x0 - x0m, x0 - x0m))
    if sxx <= 1e-18:
        return None
    b = float(np.dot(x0 - x0m, x1 - x1m) / sxx)
    a = x1m - b * x0m
    if not (0.0 < b < 1.0):                       # 발산/단위근/추세 → 평균회귀 아님
        return None
    resid = x1 - (a + b * x0)
    dof = max(1, n - 1 - 2)
    var_xi = float(resid @ resid) / dof
    if var_xi <= 1e-18:
        return None
    kappa = -math.log(b)                          # 분당
    half_life = math.log(2.0) / kappa             # 분
    m = a / (1.0 - b)
    sigma_eq = math.sqrt(var_xi / (1.0 - b * b))
    if sigma_eq <= 1e-9:
        return None
    s = (float(x[-1]) - m) / sigma_eq
    se_b = math.sqrt(var_xi / sxx)                # b 표준오차
    adf_t = (b - 1.0) / se_b if se_b > 0 else 0.0 # (b-1) t = DF 통계
    hl_ok = (p.STATARB_MIN_HALFLIFE_MIN <= half_life
             <= p.STATARB_MAX_HALFLIFE_FRAC * n)
    sig_ok = bool(adf_t <= p.STATARB_ADF_T and hl_ok)
    exp_ret = max(0.0, (abs(s) - p.STATARB_S_EXIT) * sigma_eq) if s < 0 else 0.0
    return {"s": round(s, 3), "half_life": round(half_life, 1), "kappa": round(kappa, 5),
            "sigma_eq": round(sigma_eq, 6), "adf_t": round(adf_t, 2),
            "sig_ok": sig_ok, "exp_ret": round(exp_ret, 6)}


def ou_scores(X, p=None):
    """X(W×N 누적잔차) → 열별 OU 지표를 한 번에. 위 `ou_score` 의 벡터화판.

    수학은 스칼라판과 **완전히 같다** — 종목마다 파이썬 함수를 부르던 걸 행렬 연산으로 접었을
    뿐이다. 종목 수 × 결정시점이 커지면 그 호출 오버헤드만으로 백테스트가 몇 시간이 된다
    (크립토 500종목 × 26,280시점에서 실측 3시간 → 분 단위).
    두 구현이 어긋나면 백테스트와 라이브가 달라지므로
    `crypto/test_crypto.py::test_ou_vectorized_matches_scalar` 가 원소 단위로 대조한다.
    **한쪽만 고치지 말 것.**

    반환: 유효 열 마스크 `ok` 와 지표 배열들. 무효(발산·단위근·분산 0)는 마스크에서 빠진다.
    """
    p = p or config
    W = X.shape[0]
    if W < p.STATARB_MIN_OBS:
        return None
    x0, x1 = X[:-1], X[1:]
    d0 = x0 - x0.mean(0)
    sxx = (d0 * d0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (d0 * (x1 - x1.mean(0))).sum(0) / sxx
        a = x1.mean(0) - b * x0.mean(0)
        res = x1 - (a + b * x0)
        var_xi = (res * res).sum(0) / max(1, W - 3)
        ok = (sxx > 1e-18) & (b > 0.0) & (b < 1.0) & (var_xi > 1e-18)
        b = np.where(ok, b, 0.5)                       # 무효 열은 계산만 무해하게, 마스크로 배제
        kappa = -np.log(b)
        half_life = np.log(2.0) / kappa
        m = a / (1.0 - b)
        sigma_eq = np.sqrt(var_xi / (1.0 - b * b))
        ok &= sigma_eq > 1e-9
        s = (X[-1] - m) / np.where(ok, sigma_eq, np.nan)
        se_b = np.sqrt(var_xi / sxx)
        adf_t = np.where(se_b > 0, (b - 1.0) / se_b, 0.0)
    hl_ok = ((p.STATARB_MIN_HALFLIFE_MIN <= half_life)
             & (half_life <= p.STATARB_MAX_HALFLIFE_FRAC * W))
    return {"ok": ok & np.isfinite(s), "s": s, "half_life": half_life, "kappa": kappa,
            "sigma_eq": sigma_eq, "adf_t": adf_t, "sig_ok": (adf_t <= p.STATARB_ADF_T) & hl_ok}


def market_efficiency(mret):
    """Kaufman 효율비 |Σr|/Σ|r| ∈ [0,1]. 1=완전추세(한 방향), 0=완전횡보(왕복).
    시장팩터 수익률에 적용해 '추세 레짐 vs 횡보 레짐'을 판정한다 — 역추세 전략은
    추세 레짐에서 구조적으로 손실이라 이 값이 크면 신규 진입을 멈춘다."""
    mret = np.asarray(mret, dtype=np.float64)
    denom = float(np.abs(mret).sum())
    return float(abs(mret.sum()) / denom) if denom > 1e-12 else 0.0


def recent_days(conn, n=2):
    """최근 n 영업일(오늘 포함). stat-arb 는 지금 시점까지의 롤링창을 쓰므로 오늘을 포함한다."""
    return available_days(conn, since=_lookback_since(n + 3))[-n:]


def days_for_window():
    """s-score 창(STATARB_WINDOW_MIN 분)을 담는 데 필요한 세션 수(1세션≈390분, +1 여유)."""
    return max(2, math.ceil(config.STATARB_WINDOW_MIN / 390.0) + 1)


def evaluate_scores(prices_win, markets=None, sectors=None, risky=None, lead_map=None, updated=None):
    """가격 창(DataFrame, 열=종목) → s-score 스냅샷 dict. **라이브(compute_scores)와 백테스트가 공유**
    하는 순수 평가부. prices_win 은 이미 유동성 필터된 종목만. risky=부실주 집합(없으면 게이트 무효).
    lead_map={follower:[{leader,lag,corr}]} 주면 lead-lag 확신 게이트(교집합)를 적용한다.
    파일 기록·DB 로드는 하지 않는다(호출자 책임)."""
    risky = risky or set()
    common = list(prices_win.columns)
    if not common:
        return None
    R, rc = to_residuals(prices_win, markets, sectors)
    if R.shape[0] < config.STATARB_MIN_OBS:
        return None
    W = int(min(config.STATARB_WINDOW_MIN, R.shape[0]))
    X = np.cumsum(R[-W:], axis=0)                  # W×N 누적 잔차
    last_px = prices_win.iloc[-1]
    last_ts = str(prices_win.index[-1])
    thr_edge = float(config.AUTOFOLIO_ROUND_TRIP_COST_PCT or 0.0) / 100.0

    # 시장 레짐 & 종목 절대수익률(낙하칼) — 원(raw) 수익률로 창 내에서 계산(일 경계 제거)
    idxs = list(prices_win.index)[-(W + 1):]
    Pw = prices_win.to_numpy(dtype=np.float64)[-(W + 1):]
    rr = np.diff(np.log(np.clip(Pw, 1e-9, None)), axis=0)
    sd = np.array([idxs[t][:8] == idxs[t + 1][:8] for t in range(len(idxs) - 1)])
    rr = rr[sd] if sd.any() else rr
    mret = rr.mean(axis=1) if rr.shape[0] else np.zeros(1)      # 시장 등가중 수익률
    er = market_efficiency(mret)
    mkt_cum = float(mret.sum())
    abs_by = {c: float(rr[:, i].sum()) for i, c in enumerate(common)} if rr.shape[0] else {}
    regime_trend = er > config.STATARB_MAX_TREND_ER
    regime_fall = mkt_cum < -config.STATARB_MARKET_FALL
    regime_ok = not (regime_trend or regime_fall)

    # lead-lag 확신: 선행주 최근(마지막 분) 잔차수익 × corr 합 → 후행 상승 예측값(순환 import 피해 인라인)
    confirm = {}
    if lead_map:
        recent = {rc[i]: float(R[-1, i]) for i in range(len(rc))}
        for folw, leaders in lead_map.items():
            hit = [L for L in leaders if L["leader"] in recent]
            if hit:
                confirm[folw] = sum(L["corr"] * recent[L["leader"]] for L in hit)
    require_ll = bool(config.STATARB_REQUIRE_LEADLAG and lead_map)

    V = ou_scores(X)                               # 벡터화 OU (스칼라판과 원소 단위로 동일)
    if V is None:
        return None
    scores, candidates, shorts = {}, [], []
    for i, code in enumerate(rc):
        if not V["ok"][i]:
            continue
        px = float(last_px.get(code, 0.0) or 0.0)
        if px <= 0:
            continue
        s_, sig_ = round(float(V["s"][i]), 3), round(float(V["sigma_eq"][i]), 6)
        o = {"s": s_, "half_life": round(float(V["half_life"][i]), 1),
             "kappa": round(float(V["kappa"][i]), 5), "sigma_eq": sig_,
             "adf_t": round(float(V["adf_t"][i]), 2), "sig_ok": bool(V["sig_ok"][i]),
             # 현물 롱온리라 기대반전은 저평가(s<0)에만 의미가 있다(크립토 판은 대칭)
             "exp_ret": round(max(0.0, (abs(s_) - config.STATARB_S_EXIT) * sig_), 6)
             if s_ < 0 else 0.0}
        o["price"] = px
        o["ts"] = last_ts
        o["abs_ret"] = round(abs_by.get(code, 0.0), 5)
        o["dart_risky"] = code in risky
        o["ll_confirm"] = round(confirm.get(code, 0.0), 6) if lead_map else None
        scores[code] = o
        # 매수 후보: 유의 + 과매도 + 기대반전>마찰 + 레짐 우호 + 낙하칼 아님 + 재무부실 아님
        #           + (lead-lag 있으면) 선행주가 상승 예측(교집합)
        if (o["sig_ok"] and o["s"] < -config.STATARB_S_ENTRY and o["exp_ret"] >= thr_edge
                and regime_ok and abs_by.get(code, 0.0) >= -config.STATARB_MAX_ABS_DROP
                and code not in risky
                and (not require_ll or confirm.get(code, 0.0) > config.LEADLAG_CONFIRM)):
            candidates.append({"code": code, "s": o["s"], "half_life": o["half_life"],
                               "exp_ret": o["exp_ret"], "price": px,
                               "abs_ret": round(abs_by.get(code, 0.0), 5)})
        # (가상) 숏 후보: 고평가(s>+진입)·유의. 현물 롱온리라 실거래 없음 — '롱+숏' 대비 표시용.
        elif o["sig_ok"] and o["s"] > config.STATARB_S_ENTRY:
            shorts.append({"code": code, "s": o["s"], "half_life": o["half_life"], "price": px})
    candidates.sort(key=lambda c: c["s"])          # 가장 과매도(음수 최대) 먼저
    shorts.sort(key=lambda c: -c["s"])             # 가장 고평가 먼저
    n_elig = sum(1 for v in scores.values() if v["sig_ok"])
    return {"updated": updated or mt.now_kst().isoformat(timespec="seconds"),
            "n_codes": len(rc), "window": W, "n_eligible": n_elig,
            "n_candidates": len(candidates), "scores": scores,
            "candidates": candidates[:50], "shorts": shorts[:20],
            "regime": {"er": round(er, 3), "mkt_cum": round(mkt_cum, 5),
                       "trending": bool(regime_trend), "falling": bool(regime_fall),
                       "ok": bool(regime_ok)}}


def compute_scores(conn, codes):
    """라이브 s-score 스냅샷: DB 로드 + 유동성 필터 → evaluate_scores → 캐시 기록."""
    days = recent_days(conn, days_for_window())    # 창 길이만큼 세션 로드(주 단위 창도 지원)
    if not days:
        return None
    prices = load_close_matrix(conn, codes, days=days)
    if prices.empty or prices.shape[0] < config.STATARB_MIN_OBS:
        return None
    common = list(prices.columns)
    if config.MIN_DAILY_TURNOVER_KRW > 0:         # 저유동 제외(구조적 성질 → 넉넉한 창으로 안정 추정)
        liq_days = available_days(conn, since=_lookback_since(config.BUFFER_DAYS))[-config.BUFFER_DAYS:]
        tv = avg_daily_turnover(conn, common, liq_days or days)
        liquid = [c for c in common if tv.get(c, 0.0) >= config.MIN_DAILY_TURNOVER_KRW]
        if len(liquid) >= 10:
            common = liquid
    lead_map = {}
    if config.LEADLAG_ENABLED:
        from core import leadlag                    # 지연 import(순환 회피)
        lead_map = leadlag.load_map()
    out = evaluate_scores(prices[common], crawler.load_markets(), sector_map(),
                          dart_risk.risky_set(), lead_map=lead_map)
    if out:
        try:
            config.LEADLAG_MAP_JSON.write_text(json.dumps(out, ensure_ascii=False))
        except OSError:
            pass
    return out


def run_cycle(conn, codes):
    """엔진 1사이클 = s-score 스냅샷 계산. (main.py --once dry-run 에서도 사용)
    stat-arb 은 무상태다 — s-score 는 매 분 바뀌므로 캐시·streak 추적을 두지 않는다
    (리드-랙 시절 EngineState 는 2026-07-30 제거)."""
    return compute_scores(conn, codes)


def engine_loop(map_queue, stop_event, db_path=None):
    """Engine 프로세스 본체 — 매 분 s-score 스냅샷을 큐로 내보낸다(장중에만).
    그랜저 전수스캔이 사라져 하루1회 백그라운드 재구축이 필요 없다: 매 사이클 즉시 계산."""
    from utils.logging_setup import setup
    setup("engine")
    conn = crawler.open_db(db_path)
    codes = [c for c, _ in crawler.load_universe()]
    # DART 재무위험 캐시가 오래됐으면 백그라운드로 갱신(느린 API 훑기 — 매매 루프를 막지 않는다).
    if config.DART_RISK_ENABLED and dart_risk.cache_age_hours() > config.DART_REFRESH_HOURS:
        import threading
        threading.Thread(target=dart_risk.refresh, args=(codes,), daemon=True).start()
    if config.LEADLAG_ENABLED:                      # lead-lag 맵도 일1회 백그라운드 빌드
        from core import leadlag
        if leadlag.cache_age_hours() > 20.0:
            import threading
            threading.Thread(target=lambda: leadlag.build_map(crawler.open_db(db_path), codes),
                             daemon=True).start()
    if config.SECTOR_SOURCE == "cluster":           # 상관 클러스터도 일1회 (계층 클러스터링은 느리다)
        from core import clusters
        if clusters.cache_age_hours() > config.CLUSTER_REFRESH_HOURS:
            import threading
            threading.Thread(target=lambda: clusters.build_map(crawler.open_db(db_path), codes),
                             daemon=True).start()
    while not stop_event.is_set():
        if not mt.in_crawl_session(mt.now_kst()):
            if stop_event.wait(30):
                break
            continue
        t0 = time.monotonic()
        try:
            out = compute_scores(conn, codes)
            if out:
                while not map_queue.empty():       # 최신 스냅샷만 유지
                    try:
                        map_queue.get_nowait()
                    except Exception:              # noqa: BLE001
                        break
                map_queue.put(out)
                log.info("s-score 스냅샷: %d종목 · 적격 %d · 매수후보 %d (%.2fs)",
                         out["n_codes"], out["n_eligible"], out["n_candidates"],
                         time.monotonic() - t0)
        except Exception as e:                     # 연산 실패도 루프를 죽이지 않음
            log.error("engine cycle 예외: %s", e, exc_info=True)
        wait = max(5.0, config.ENGINE_INTERVAL_SEC - (time.monotonic() - t0))
        if stop_event.wait(wait):
            break
    conn.close()
