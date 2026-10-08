"""팩터 모델 — 고유포트폴리오 잔차화 · RMT 상관행렬 정제 · 상관 클러스터링.

Avellaneda & Lee (2010) §2.2. **자산군에 무관한 순수 수학**이라 KRX(core/)와
크립토(crypto/)가 같은 파일을 쓴다 — numpy 말고는 아무것도 import 하지 않는다.

원래 KRX 는 KOSPI·KOSDAQ 등가중평균 2팩터 + 거래소 업종분류를 썼는데,
크립토엔 지수도 업종분류도 없어서 고유포트폴리오와 데이터 기반 클러스터로 대체했다.
2026-08-07 사장 지시로 그 로직을 KRX 쪽에도 그대로 적용했다.

클러스터링과 s-score 엔진이 **같은 잔차 정의**를 쓴다. 이게 어긋나면 클러스터가 잡아낸
공통성분과 엔진이 제거하는 공통성분이 달라져, 클러스터 디민이 오히려 노이즈를 넣는다.
"""
from __future__ import annotations

import numpy as np


def eigen_factors(R: np.ndarray, k: int = 2):
    """R(T×N 로그수익률) → (F: T×k 팩터수익률, expl: 상위 k 고유값 설명비율).

    표준화 수익률 Y=(R-μ)/σ 의 상관행렬 고유벡터 v 로 고유포트폴리오를 만든다:
    가중치 Q_ji = v_ji/σ_i, 팩터수익 F_j(t)=Σ_i Q_ji·R_i(t).
    변동성으로 나누는 게 핵심 — 안 나누면 고변동 밈코인 몇 개가 PC1 을 통째로 차지한다.
    """
    T, N = R.shape
    if k <= 0:                       # 팩터 제거 안 함 — 원 수익률 상관을 그대로 쓰고 싶을 때
        return np.zeros((T, 0)), np.zeros(0)
    k = max(1, min(k, N - 1, T - 1))
    sd = R.std(axis=0)
    sd = np.where(sd > 1e-12, sd, np.nan)
    Y = (R - R.mean(axis=0)) / sd
    ok = np.isfinite(Y).all(axis=0)
    if ok.sum() < k + 1:
        return np.zeros((T, 0)), np.zeros(0)
    C = np.corrcoef(Y[:, ok], rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:k]
    Q = np.zeros((N, k))
    Q[ok] = V[:, idx] / sd[ok, None]
    F = R @ Q
    return F, w[idx] / max(1e-12, w.sum())


def residualize(R: np.ndarray, k: int = 2, labels=None):
    """R(T×N) → 팩터중립 잔차 ε(T×N).

    1) 상위 k 고유포트폴리오 회귀 잔차 (= 시장 모드 제거)
    2) labels 주면 **클러스터 동시점 평균 잔차**를 추가로 뺀다 — KRX 판의 섹터 디민과
       같은 자리다. 클러스터가 통째로 눌릴 때 개별 저평가로 오인하지 않게 한다.
    """
    Rc = R - R.mean(axis=0)
    F, _ = eigen_factors(R, k)
    if F.shape[1]:
        Fc = F - F.mean(axis=0)
        try:
            B = np.linalg.solve(Fc.T @ Fc + 1e-12 * np.eye(Fc.shape[1]), Fc.T @ Rc)
            Rc = Rc - Fc @ B
        except np.linalg.LinAlgError:
            pass
    if labels is not None:
        labels = np.asarray(labels)
        for g in np.unique(labels[labels >= 0]):
            m = labels == g
            if m.sum() >= 2:
                Rc[:, m] -= Rc[:, m].mean(axis=1, keepdims=True)
    return Rc


def rmt_clip(C: np.ndarray, T: int) -> np.ndarray:
    """Marchenko-Pastur 잡음 고유값 클리핑 (Laloux et al. 1999).

    N 종목 × T 관측이면 순수 잡음만으로도 고유값이 (1±√(N/T))² 까지 퍼진다.
    N=800·T=4000 이면 상한 2.0 — 그 아래 고유값은 정보가 아니라 표본잡음이다.
    잡음 대역 고유값을 평균으로 눕히고(대각합 보존) 대각을 1 로 재정규화한다.
    이걸 안 하면 클러스터가 매 창마다 흔들린다.
    """
    N = C.shape[0]
    if N < 2 or T < 2:
        return C
    w, V = np.linalg.eigh(C)
    lam_max = (1.0 + np.sqrt(N / T)) ** 2
    noise = w < lam_max
    if noise.any():
        # 전부 잡음이면 전부 눕힌다 → 단위행렬. 그게 정답이다(구조가 없다는 뜻).
        w = w.copy()
        w[noise] = w[noise].mean()
    Cf = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(Cf), 1e-12, None))
    Cf = Cf / np.outer(d, d)
    np.fill_diagonal(Cf, 1.0)
    return np.clip(Cf, -1.0, 1.0)


def corr_linkage(R: np.ndarray, n_factors: int = 2):
    """수익률 행렬(T×N) → (정제된 잔차상관 C, 계층 연결 Z).

    잔차화 → 상관 → RMT 클리핑 → d=√(2(1−ρ)) average linkage. 거리는 진짜 metric 이다
    (Mantegna 1999). 여기 한 곳에만 두는 이유는 **클러스터를 만든 잔차 정의와 엔진이 쓰는
    잔차 정의가 같아야** 하기 때문이다 — 갈리면 클러스터 디민이 노이즈를 넣는 꼴이 된다.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    C = rmt_clip(np.nan_to_num(np.corrcoef(residualize(R, n_factors), rowvar=False), nan=0.0),
                 R.shape[0])
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(D, 0.0)
    return C, linkage(squareform((D + D.T) / 2, checks=False), method="average")


def cluster_labels(R: np.ndarray, k: int, n_factors: int = 2) -> np.ndarray:
    """수익률 행렬(T×N) → 1-기반 클러스터 라벨(N,). corr_linkage 의 단일 k 판."""
    from scipy.cluster.hierarchy import fcluster

    n = R.shape[1]
    if n < 3:
        return np.ones(n, dtype=int)
    return fcluster(corr_linkage(R, n_factors)[1], max(2, min(k, n - 1)), criterion="maxclust")
