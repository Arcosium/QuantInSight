"""크립토 stat-arb 자체점검 — 네트워크·아카이브 없이 돈다.

  python3 -m crypto.test_crypto      또는  python3 -m pytest crypto/ -q

여기서 지키는 것은 넷이다: 잡음 제거가 실제로 되는가, 팩터가 실제로 빠지는가,
과매도/과매수 판정 부호가 맞는가, **미래를 안 보는가**. 나머지는 백테스트가 잰다.
"""
import copy

import numpy as np
import pandas as pd

from core import factors
from crypto import backtest, cluster, engine, screen

RNG = np.random.default_rng(7)


def test_rmt_clip_kills_pure_noise():
    """순수 잡음 상관행렬은 클리핑 후 거의 단위행렬이 돼야 한다.

    N=120·T=400 이면 표본잡음만으로도 |ρ|가 0.15 넘는 쌍이 수백 개 나온다.
    그걸 그대로 클러스터링하면 매 창마다 다른 '섹터'가 발견된다.
    """
    N, T = 120, 400
    R = RNG.normal(size=(T, N))
    C = np.corrcoef(R, rowvar=False)
    off = ~np.eye(N, dtype=bool)
    before = np.abs(C[off]).max()
    after = np.abs(factors.rmt_clip(C, T)[off]).max()
    assert before > 0.15, before                 # 잡음만으로도 이만큼 나온다
    assert after < before / 3, (before, after)


def test_residualize_removes_common_factor():
    """공통 팩터에 실린 성분은 잔차에서 사라져야 한다 — 안 그러면 전 종목이 한 덩어리로 뭉친다."""
    T, N = 500, 40
    f = RNG.normal(size=T) * 0.02
    beta = RNG.uniform(0.5, 1.5, N)
    R = np.outer(f, beta) + RNG.normal(size=(T, N)) * 0.002
    raw = np.array([np.corrcoef(R[:, i], f)[0, 1] for i in range(N)])
    res = factors.residualize(R, k=1)
    out = np.array([np.corrcoef(res[:, i], f)[0, 1] for i in range(N)])
    assert np.abs(raw).mean() > 0.9, np.abs(raw).mean()
    assert np.abs(out).mean() < 0.1, np.abs(out).mean()


def test_residualize_cluster_demean():
    """클러스터 동시점 평균이 빠지면 같은 클러스터 잔차의 합은 0 이다."""
    R = RNG.normal(size=(200, 6)) * 0.01
    res = factors.residualize(R, k=1, labels=np.array([1, 1, 1, 2, 2, 2]))
    assert np.abs(res[:, :3].sum(axis=1)).max() < 1e-9
    assert np.abs(res[:, 3:].sum(axis=1)).max() < 1e-9


def test_corwin_schultz_ranks_spread():
    """호가 스프레드가 큰 종목이 더 큰 추정치를 받아야 한다(절대값이 아니라 순위가 쓸모)."""
    T = 600
    mid = 100 * np.exp(np.cumsum(RNG.normal(size=T) * 0.001))
    cols, hi, lo = [], [], []
    for spread in (0.0, 0.002, 0.010):
        half = mid * spread / 2
        # 봉 내부 변동 + 호가 튐: 고가는 ask 쪽, 저가는 bid 쪽에서 찍힌다
        rng = mid * 0.001
        hi.append(mid + rng + half)
        lo.append(mid - rng - half)
        cols.append(f"s{spread}")
    H = pd.DataFrame(np.column_stack(hi), columns=cols)
    L = pd.DataFrame(np.column_stack(lo), columns=cols)
    est = screen.corwin_schultz(H, L)
    assert est.iloc[0] < est.iloc[1] < est.iloc[2], est.to_dict()


def test_wash_z_flags_no_impact_volume():
    """거래대금은 큰데 가격이 안 밀리는 종목을 잡아야 하고, 대형주는 안 잡아야 한다.

    2차항이 없으면 유동성의 초선형성 때문에 최대형(=여기선 MEGA)이 오히려 상위로 올라온다.
    실제로 1차 적합에서 BTC·ETH 가 탈락선을 넘었던 회귀 방지 테스트다.
    """
    n = 200
    adv = np.exp(RNG.uniform(np.log(2e6), np.log(2e9), n))
    amihud = 1e6 / adv ** 0.75 * np.exp(RNG.normal(0, 0.25, n))   # 초선형 유동성
    s = pd.DataFrame({"adv_usd": adv, "amihud": amihud,
                      "vol": np.full(n, 0.8)}, index=[f"C{i}" for i in range(n)])
    s.loc["WASH"] = {"adv_usd": 5e7, "amihud": 1e6 / 5e7 ** 0.75 / 50, "vol": 0.8}
    s.loc["MEGA"] = {"adv_usd": 2e10, "amihud": 1e6 / 2e10 ** 0.75, "vol": 0.8}
    z = screen._wash_z(s, 1e6, 0.2)
    assert z["WASH"] > 2.5, z["WASH"]
    assert abs(z["MEGA"]) < 2.0, z["MEGA"]


def _ou_path(T, b, sigma, rng):
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = b * x[t - 1] + rng.normal() * sigma
    return x


def test_snapshot_signs_and_dollar_neutral():
    """과매도는 longs, 과매수는 shorts. 그리고 양다리 수가 같아야 한다(달러중립).

    ⚠ 종목 수를 적게 잡으면 이 테스트가 **거짓으로 실패한다**. 주입한 이탈 자체가 상위
    고유포트폴리오가 돼 팩터 회귀에 흡수되기 때문이다(N=12·k=2 에서 실제로 겪었다 —
    s 부호가 뒤집혀 나왔다). 팩터 하나가 개별 종목에 먹히지 않을 만큼 N 을 크게 둔다.
    """
    T, N = 260, 60
    rng = np.random.default_rng(11)
    f = np.cumsum(rng.normal(size=T) * 0.01)                     # 공통 시장 모드
    logp = f[:, None] + rng.normal(size=(T, N)) * 0.0005 + np.log(100)
    # 마지막 값이 가장 깊이 내려간 OU 경로를 고른다 — 그게 '지금 과매도'인 종목이다
    dev = min((_ou_path(T, 0.95, 0.01, rng) for _ in range(200)),
              key=lambda x: x[-1] / (x.std() or 1))
    logp[:, 0] += dev
    logp[:, 1] -= dev                                            # 대칭으로 들린 종목
    px = pd.DataFrame(np.exp(logp), columns=[f"A{i}" for i in range(N)],
                      index=np.arange(T) * 900_000)
    p = copy.copy(engine.P)
    p.FEE_RT, p.DEFAULT_SPREAD, p.STATARB_MIN_OBS, p.N_FACTORS = 0.0, 0.0, 60, 1
    snap = engine.snapshot(px, p=p)
    assert snap is not None
    s0, s1 = snap["scores"]["A0"]["s"], snap["scores"]["A1"]["s"]
    assert s0 < -p.STATARB_S_ENTRY, s0
    assert s1 > p.STATARB_S_ENTRY, s1
    assert "A0" in [o["base"] for o in snap["longs"]], snap["longs"]
    assert "A1" in [o["base"] for o in snap["shorts"]], snap["shorts"]
    assert len(snap["longs"]) == len(snap["shorts"])   # 달러중립


def test_snapshot_ignores_rows_after_the_window():
    """창 뒤에 무슨 일이 있어도 스냅샷이 바뀌면 안 된다 — lookahead 방지의 최소 조건."""
    T, N = 200, 10
    px = pd.DataFrame(np.exp(np.cumsum(RNG.normal(size=(T, N)) * 0.01, axis=0)) * 50,
                      columns=[f"A{i}" for i in range(N)], index=np.arange(T) * 900_000)
    a = engine.snapshot(px.iloc[:150])
    future = px.copy()
    future.iloc[150:] *= 5.0                        # 미래를 통째로 뒤흔든다
    b = engine.snapshot(future.iloc[:150])
    assert a is not None and json_eq(a["scores"], b["scores"])


def json_eq(x, y):
    import json
    return json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)


def test_backtest_refit_never_sees_the_present(monkeypatch=None):
    """유니버스 재적합은 결정시점 **이전** 봉만 받아야 한다. 여기가 새면 전부 무효다."""
    ts = np.arange(0, 400 * 3_600_000, 3_600_000)
    panel = pd.DataFrame({"ts": np.repeat(ts, 3), "base": ["A", "B", "C"] * len(ts)})
    seen = {}

    def spy(sl, *a, **k):
        seen["max_ts"] = int(sl["ts"].max())
        return None
    orig = cluster.fit
    cluster.fit = spy
    try:
        upto = int(ts[300])
        backtest._fit_universe(panel, upto, win_days=5, k=10, n_factors=2)
    finally:
        cluster.fit = orig
    assert seen["max_ts"] < upto, (seen["max_ts"], upto)


def test_windows_covers_the_tail():
    """마지막 창은 데이터 끝에 붙어야 한다 — 안 그러면 최신 3개월이 조용히 빠진다."""
    day = 86_400_000
    w = cluster.windows(0, 500 * day, 180, 90)
    assert w[0][0] == 0
    assert w[-1][1] == 500 * day, w[-1]
    assert all(b - a == 180 * day for a, b in w)


def test_ou_vectorized_matches_scalar():
    """벡터화판이 스칼라판과 원소 단위로 같아야 한다.

    백테스트 속도 때문에 OU 적합을 행렬로 접었는데(500종목 루프가 3시간짜리 병목이었다),
    두 구현이 갈리면 백테스트가 재는 전략과 라이브가 도는 전략이 달라진다.
    유효/무효 판정(발산·단위근·분산 0)까지 일치시킨다.
    """
    from core.analytics import ou_score
    p = copy.copy(engine.P)
    p.STATARB_MIN_OBS = 60
    cols = [np.cumsum(RNG.normal(size=120) * 0.01),                       # 랜덤워크(대개 무효)
            np.zeros(120),                                                # 분산 0 → 무효
            np.cumsum(RNG.normal(size=120)) * 0 + np.arange(120) * 0.01]  # 순수 추세 → 무효
    for _ in range(12):                                                   # 진짜 OU 표본
        x, v = [0.0], 0.0
        for _ in range(120):
            v = 0.93 * v + RNG.normal() * 0.01
            x.append(v)
        cols.append(np.array(x[1:]))
    X = np.column_stack(cols)
    V = engine.ou_scores(X, p)
    n_valid = 0
    for i in range(X.shape[1]):
        o = ou_score(X[:, i], p)
        assert bool(V["ok"][i]) == (o is not None), i
        if o is None:
            continue
        n_valid += 1
        for key, nd in (("s", 3), ("half_life", 1), ("sigma_eq", 6), ("adf_t", 2)):
            assert round(float(V[key][i]), nd) == o[key], (i, key, V[key][i], o[key])
        assert bool(V["sig_ok"][i]) == o["sig_ok"], i
    assert n_valid >= 8, n_valid          # 표본이 전부 무효면 아무것도 검증 못 한 것


def test_ou_params_are_injectable():
    """core.analytics.ou_score 가 크립토 파라미터를 받는가(KRX 기본값은 그대로인가)."""
    import config
    from core.analytics import ou_score
    x = np.cumsum(RNG.normal(size=80) * 0.01)
    assert ou_score(x) is None                      # KRX 기본 MIN_OBS=90 → 미달
    p = copy.copy(engine.P)
    p.STATARB_MIN_OBS = 50
    assert ou_score(x, p) is not None
    assert config.STATARB_MIN_OBS == 90             # 기본값 오염 없음


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
