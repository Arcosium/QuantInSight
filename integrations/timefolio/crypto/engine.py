"""s-score 엔진 — 크립토 perp 롱숏 통계적 차익거래.

KRX 판(core/analytics.compute_scores)과 같은 수학, 다른 세 가지:
  · 24/7 이라 세션·동시호가·일 경계가 없다 (밤샘 갭 처리 자체가 불필요)
  · 팩터가 지수가 아니라 **고유포트폴리오**다 (crypto/factors.py)
  · perp 이라 **숏 다리를 실제로 쓴다** — 논문 그대로의 달러중립 롱숏.
    KRX 는 현물이라 롱만 취했고, 그래서 시장 하락에 통째로 노출됐다.

시간 단위는 전부 '봉'이다(분 아님). BAR_MIN 을 바꾸면 반감기 문턱도 같이 따라간다.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core import factors
from core.analytics import market_efficiency, ou_scores    # noqa: F401 (engine.ou_scores 로 노출)

log = logging.getLogger("crypto.engine")


def _e(k, d, cast=float):
    v = os.environ.get(k)
    try:
        return cast(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


# 봉 단위 파라미터. KRX config 와 섞지 않는다 — 라이브 KRX 프로세스를 건드리지 않기 위함이다.
P = SimpleNamespace(
    BAR_MIN=_e("CRY_BAR_MIN", 15, int),              # 신호 봉 주기(분)
    WINDOW_BARS=_e("CRY_WINDOW", 96, int),           # 잔차 누적·OU 창 = 15분×96 = 24시간
    N_FACTORS=_e("CRY_FACTORS", 2, int),             # 제거할 고유포트폴리오 수
    STATARB_MIN_OBS=_e("CRY_MIN_OBS", 60, int),
    STATARB_S_ENTRY=_e("CRY_S_ENTRY", 1.25),
    STATARB_S_EXIT=_e("CRY_S_EXIT", 0.5),
    STATARB_S_STOP=_e("CRY_S_STOP", 3.0),
    STATARB_MIN_HALFLIFE_MIN=_e("CRY_MIN_HL", 3.0),  # 봉. 이하 = 마이크로구조 노이즈
    STATARB_MAX_HALFLIFE_FRAC=_e("CRY_MAX_HL_FRAC", 0.5),
    STATARB_ADF_T=_e("CRY_ADF_T", -2.0),
    FEE_RT=_e("CRY_FEE_RT", 0.0010),                 # 왕복 수수료(바이낸스 USDⓈ-M 테이커 0.05%×2)
    SLIP_RT=_e("CRY_SLIP_RT", 0.0002),               # 왕복 슬리피지 가정(2bp)
    MAX_POSITIONS=_e("CRY_MAX_POS", 20, int),        # 다리 하나당 최대 종목 수
)


def cost_of(base: str, spreads=None, p=P) -> float:
    """왕복 마찰 = 수수료 + 슬리피지. 전 종목 동일하게 잡는다.

    **cs_spread 를 비용으로 쓰지 않는다.** Corwin-Schultz 는 일봉용 추정량이라 1시간봉에서는
    봉 내부 변동성이 스프레드를 압도해 크게 과대추정한다 — 실측 BTC 9.7bp 인데 바이낸스
    BTC perp 실제 호가차는 0.5~1bp 로 10배 넘게 부풀었다. 그대로 비용에 넣었더니 왕복 31bp 가
    돼서 총이익 3.5bp/거래를 통째로 잡아먹었다(2026-08-07). cs_spread 는 유동성 **순위**용
    (screen.passes)으로만 쓴다.

    종목별 실비용을 쓰려면 호가창 데이터가 필요하다 — MultiEX capture lane 의 몫이고,
    CryptoBars 의 OHLCV 로는 원리상 나오지 않는다. spreads 인자는 호환용으로 남긴다.
    """
    return p.FEE_RT + p.SLIP_RT


def snapshot(px: pd.DataFrame, labels=None, spreads=None, updated=None, p=P) -> dict | None:
    """가격 창(행=ts, 열=base) → s-score 스냅샷. **라이브와 백테스트가 공유하는 순수 평가부.**

    labels: px.columns 순서의 클러스터 라벨(없으면 클러스터 디민 생략).
    미래 데이터를 절대 보지 않는다 — 창의 마지막 행이 '지금'이다.
    """
    if px is None or px.shape[1] < 5:
        return None
    px = px.ffill().bfill()
    R = np.log(px.to_numpy(np.float64))
    R = np.diff(R, axis=0)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    if R.shape[0] < p.STATARB_MIN_OBS:
        return None

    resid = factors.residualize(R, p.N_FACTORS, labels)
    W = int(min(p.WINDOW_BARS, resid.shape[0]))
    X = np.cumsum(resid[-W:], axis=0)                 # W×N 누적 잔차
    bases = list(px.columns)
    last_px = px.iloc[-1]
    mret = R[-W:].mean(axis=1)

    V = ou_scores(X, p)
    if V is None:
        return None
    prices = pd.to_numeric(last_px, errors="coerce").to_numpy(np.float64)

    scores, longs, shorts = {}, [], []
    for i, b in enumerate(bases):
        if not V["ok"][i] or not (prices[i] > 0):
            continue
        s, sig = round(float(V["s"][i]), 3), round(float(V["sigma_eq"][i]), 6)
        o = {"s": s, "half_life": round(float(V["half_life"][i]), 1),
             "kappa": round(float(V["kappa"][i]), 5), "sigma_eq": sig,
             "adf_t": round(float(V["adf_t"][i]), 2), "sig_ok": bool(V["sig_ok"][i]),
             # 기대 반전폭: |s| 가 청산문턱까지 되돌아올 때 먹는 잔차 폭. 롱·숏 대칭.
             "exp_ret": round(max(0.0, (abs(s) - p.STATARB_S_EXIT) * sig), 6),
             "price": float(prices[i]), "base": b,
             "cost": round(cost_of(b, spreads, p), 6)}
        scores[b] = o
        if not (o["sig_ok"] and o["exp_ret"] >= o["cost"]):
            continue
        if s < -p.STATARB_S_ENTRY:
            longs.append(o)
        elif s > p.STATARB_S_ENTRY:
            shorts.append(o)

    longs.sort(key=lambda o: o["s"])                  # 가장 과매도 먼저
    shorts.sort(key=lambda o: -o["s"])                # 가장 과매수 먼저
    n = min(len(longs), len(shorts), p.MAX_POSITIONS)  # 달러중립: 양다리 같은 수만
    return {"updated": updated, "n_codes": len(bases), "window": W,
            "n_eligible": sum(1 for v in scores.values() if v["sig_ok"]),
            "scores": scores, "longs": longs[:n], "shorts": shorts[:n],
            "n_longs_raw": len(longs), "n_shorts_raw": len(shorts),
            "regime": {"er": round(market_efficiency(mret), 3),
                       "mkt_cum": round(float(mret.sum()), 5)}}
