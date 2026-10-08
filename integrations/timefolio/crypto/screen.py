"""유동성·가짜유동성 스크리닝 — 페어 후보로 올리기 전에 거를 것들.

팀 논의(2026-08-07)에서 나온 두 문제를 겨냥한다:
  ① 거래량 많고 시총 낮은 자산에 워시트레이딩 신호가 많다
  ② 극소형은 마켓메이커가 가격을 지배한다

거래대금/시총 비율은 이 둘을 못 거른다 — **거래량은 만들 수 있어도 가격충격은 만들기 어렵다**.
그래서 충격 기반 지표를 쓴다:

  amihud   = mean(|r| / 거래대금)  ─ 1달러가 가격을 얼마나 미는가
  wash_z   = log(amihud) 를 log(거래대금)에 회귀한 잔차의 **음수 방향**
             = "거래대금이 이 정도면 이만큼 밀려야 하는데 안 밀린다" → 자기거래 의심
  cs_spread= Corwin-Schultz(2012) 고저가 스프레드 추정 ─ OHLC 만으로 진짜 왕복비용
  zero_vol = 거래대금이 있는데 가격이 안 움직인 봉의 비율

⚠ wash_z 는 **이상 신호이지 증거가 아니다.** 진짜 워시트레이딩과 "마켓메이커가 촘촘히
호가를 대서 충격이 작은 우량 종목"이 같은 방향으로 찍힌다. 그래서 거래대금 하한을
먼저 걸고 **그 안에서의 상대 순위**로만 쓴다(BTC 가 잘려나가지 않게).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_CS = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame) -> pd.Series:
    """연속 두 봉의 고저가 → 유효 스프레드(비율) 추정. 종목별 중앙값.

    Corwin & Schultz (2012), JF 67(2). 고가-저가 폭은 변동성과 스프레드를 함께 담는데,
    2봉을 합치면 변동성은 시간에 비례하고 스프레드는 안 늘어난다 — 그 차이로 분리한다.
    음수 추정치는 0 으로 본다(논문 권고).
    """
    h, l = high.to_numpy(float), low.to_numpy(float)
    ok = (h > 0) & (l > 0)
    hl = np.where(ok, np.log(np.divide(h, l, out=np.ones_like(h), where=ok)), np.nan)
    beta = hl[:-1] ** 2 + hl[1:] ** 2
    h2 = np.fmax(h[:-1], h[1:])
    l2 = np.fmin(l[:-1], l[1:])
    good = (h2 > 0) & (l2 > 0)
    gamma = np.where(good, np.log(np.divide(h2, l2, out=np.ones_like(h2), where=good)) ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
    return pd.Series(np.nanmedian(s, axis=0), index=high.columns)


def stats(panel: pd.DataFrame, freq_min: int, min_adv=1e6, min_vol=0.2) -> pd.DataFrame:
    """롱포맷 패널(한 창) → base 별 유동성·마찰·가짜유동성 지표표.
    min_adv·min_vol 은 **wash_z 회귀를 적합할 표본**을 정하는 데만 쓴다(필터는 passes 담당)."""
    from crypto import data as D

    px = D.pivot(panel, "close")
    qv = D.pivot(panel, "qv").reindex_like(px)
    nb = D.pivot(panel, "n").reindex_like(px)
    hi, lo = D.pivot(panel, "high").reindex_like(px), D.pivot(panel, "low").reindex_like(px)

    r = D.log_returns(px)
    qv_r = qv.iloc[1:]                                   # 수익률과 행 맞추기
    days = max(1.0, (px.index[-1] - px.index[0]) / 86_400_000)

    dollar = qv.sum(min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        impact = (r.abs() / qv_r.where(qv_r > 0)).replace([np.inf, -np.inf], np.nan)
    traded = (qv_r > 0) & (nb.iloc[1:] > 0)

    out = pd.DataFrame(index=px.columns)
    out["cover"] = px.notna().mean()                     # 창 내 관측 비율(신규·상폐 판별)
    out["bars"] = nb.sum(min_count=1)
    out["adv_usd"] = dollar / days
    out["amihud"] = impact.mean() * 1e9                  # 10억달러당 로그수익률
    out["cs_spread"] = corwin_schultz(hi, lo)
    out["zero_vol"] = ((r.abs() < 1e-12) & traded).sum() / traded.sum().replace(0, np.nan)
    out["vol"] = r.std() * np.sqrt(365 * 24 * 60 / freq_min)   # 연율 변동성
    out["ret"] = np.log(px.ffill().iloc[-1] / px.ffill().iloc[0])
    out["wash_z"] = _wash_z(out, min_adv, min_vol)
    return out


def _wash_z(s: pd.DataFrame, min_adv: float, min_vol: float) -> pd.Series:
    """log(amihud) ~ log(adv) 회귀 잔차를 부호 뒤집어 표준화. 클수록 '충격 없는 거래량'.

    **회귀는 거래대금 하한을 통과한 종목으로만 적합한다.** 처음엔 전 종목으로 적합했는데,
    일 거래대금 $1M 미만 구간에서는 |r| 도 함께 작아져 amihud~adv 관계가 아예 평평해진다
    (2026-08-07 실측 — 하위 구간이 기울기를 지배해 BTC·ETH·XAU 가 워시 상위로 올라왔다).
    하한 아래 종목은 어차피 adv 필터가 거르므로 NaN 으로 두고 통과 처리한다.

    **2차항이 필수다.** 유동성은 규모에 초선형이라(호가가 깊어지고 마켓메이커가 늘어난다)
    직선으로 맞추면 최상위가 통째로 "충격이 너무 작다"로 찍힌다 — 1차 적합에서
    BTC z=2.41 · ETH 2.15 로 둘 다 탈락선을 넘었다. 2차항을 넣으면 BTC −0.41 · ETH −0.44 로
    제자리를 찾고 상위는 TRX·XAUT·PAXG·COPPER·GRVT 로 바뀐다(2026-08-07 실측, R² 0.66→0.68).
    Huber 손실은 진짜 이상치가 기준선을 자기 쪽으로 끌어당기는 것을 막는다.

    실제로 잡히는 건 "거래대금에 비해 가격이 안 움직이는 자산"이고, 여기엔 워시트레이딩뿐
    아니라 **페그 자산(XAUT·PAXG 토큰화 금)과 마켓메이커 지배 종목**이 함께 들어온다.
    평균회귀 페어북에 넣고 싶지 않은 건 어느 쪽이든 마찬가지라 필터로는 유효하지만,
    이 지표를 "워시트레이딩 증거"로 인용하면 안 된다.
    """
    z = pd.Series(np.nan, index=s.index)
    m = ((s["adv_usd"] >= min_adv) & (s["vol"] >= min_vol)
         & (s["amihud"] > 0) & np.isfinite(s["amihud"]))
    if m.sum() < 30:
        return z
    lx = np.log(s.loc[m, "adv_usd"].to_numpy())
    X = np.column_stack([lx, lx ** 2])
    y = np.log(s.loc[m, "amihud"].to_numpy())
    from sklearn.linear_model import HuberRegressor
    res = y - HuberRegressor().fit(X, y).predict(X)
    sd = res.std()
    z[m] = -res / sd if sd > 1e-12 else 0.0
    return z


def passes(s: pd.DataFrame, min_adv=1e6, max_spread=0.004, max_zero=0.35,
           max_wash_z=2.0, min_cover=0.95, min_vol=0.2) -> pd.Series:
    """1차 필터. 기본값은 실측 후 조정할 자리표시자다 — `--report` 로 분포를 먼저 볼 것.

    max_spread 0.4% = 왕복 마찰이 이미 s-score 기대반전폭을 먹는 수준.
    min_vol 20% 연율 = 스테이블코인·페그 자산 제거(평균회귀가 아니라 그냥 안 움직임).
    """
    return (s["cover"] >= min_cover) & (s["adv_usd"] >= min_adv) \
        & (s["cs_spread"].fillna(1.0) <= max_spread) \
        & (s["zero_vol"].fillna(1.0) <= max_zero) \
        & (s["wash_z"].fillna(0.0) <= max_wash_z) \
        & (s["vol"].fillna(0.0) >= min_vol)
