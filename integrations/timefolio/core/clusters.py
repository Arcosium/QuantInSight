"""KRX 상관 클러스터 — 거래소 업종분류 대신 **가격이 말하는 섹터**를 쓴다.

2026-08-07 사장 지시로 크립토 쪽 로직을 그대로 옮겨왔다. 원래 `_sector_demean` 은
universe.csv 의 `sector` 컬럼(「반도체와반도체장비」같은 거래소 업종)을 썼는데, 그건
사업 내용 분류이지 **같이 움직이는가**의 분류가 아니다. 조선주가 방산과 함께 움직이거나
2차전지가 통째로 도는 국면을 업종표는 못 잡는다.

crypto 판과 완전히 같은 절차다(`core/factors.corr_linkage`): 고유포트폴리오 잔차화 →
RMT 클리핑 → d=√(2(1−ρ)) 계층 클러스터링. 하루 1회 백그라운드로 지어 캐시한다
(lead-lag 맵과 같은 방식) — 매 분 다시 지을 이유가 없고, 매 분 라벨이 흔들리면
섹터 디민이 오히려 노이즈가 된다.

크립토와 다른 점 하나: KRX bars.db 에는 고가·저가가 없어 Corwin-Schultz 스프레드 추정이
불가능하다. 유동성 필터는 기존 `MIN_DAILY_TURNOVER_KRW` 가 그대로 담당한다.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np

import config
from core import factors

log = logging.getLogger("lag.clusters")
CACHE = config.DATA_DIR / "clusters_kr.json"


def cache_age_hours() -> float:
    if not CACHE.exists():
        return 1e9
    return (time.time() - CACHE.stat().st_mtime) / 3600.0


def load_map() -> dict:
    """{code: '군12'} 또는 {}. 캐시가 없으면 빈 dict → 섹터 디민이 자동으로 꺼진다."""
    try:
        return json.loads(CACHE.read_text()).get("labels", {})
    except (OSError, ValueError):
        return {}


def build_map(conn, codes, upto_day=None) -> dict:
    """학습창 분봉으로 클러스터를 지어 캐시에 쓰고 반환한다.

    upto_day 를 주면 **그날 이전** 데이터만 쓴다(백테스트 무-lookahead). 라이브는 None.
    """
    from core import analytics as A

    days = A.available_days(conn, since=A._lookback_since(config.BUFFER_DAYS + 5))
    if upto_day:
        days = [d for d in days if d < upto_day]
    days = days[-config.BUFFER_DAYS:]
    if len(days) < 3:
        log.warning("클러스터 학습 영업일 부족(%d)", len(days))
        return {}
    prices = A.load_close_matrix(conn, codes, days=days)
    if prices.empty or prices.shape[1] < 10:
        return {}

    # 잔차화는 factors 가 하므로 여기서는 원 수익률만 만든다. 일 경계 행은 제거한다
    # (밤샘 갭은 종목 간 공통이라 상관을 통째로 부풀린다 — KRX 판의 오래된 규칙).
    P = prices.to_numpy(dtype=np.float64)
    ts = list(prices.index)
    same_day = np.array([ts[i][:8] == ts[i + 1][:8] for i in range(len(ts) - 1)])
    R = np.nan_to_num(np.log(P[1:] / P[:-1])[same_day], nan=0.0, posinf=0.0, neginf=0.0)
    if R.shape[0] < 100:
        return {}

    k = min(config.CLUSTER_K, max(2, R.shape[1] // 3))
    lab = factors.cluster_labels(R, k, config.CLUSTER_FACTORS)
    labels = {c: f"군{int(g)}" for c, g in zip(prices.columns, lab)}
    sizes = np.bincount(lab)[1:]
    out = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "days": f"{days[0]}~{days[-1]}",
           "n_codes": len(labels), "k": int(k),
           "size_p50": int(np.median(sizes[sizes > 0])),
           "size_max": int(sizes.max()), "labels": labels}
    try:
        CACHE.write_text(json.dumps(out, ensure_ascii=False))
    except OSError:
        pass
    log.info("클러스터 %d개 · %d종목 · 창 %s (중앙 크기 %d)",
             out["k"], out["n_codes"], out["days"], out["size_p50"])
    return labels


if __name__ == "__main__":                       # python3 -m core.clusters
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from core import crawler
    conn = crawler.open_db()
    codes = [c for c, _ in crawler.load_universe()]
    labels = build_map(conn, codes)
    groups = {}
    for code, g in labels.items():
        groups.setdefault(g, []).append(code)
    names = dict(crawler.load_universe())
    for g, mem in sorted(groups.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"{g:<6}({len(mem):>3})  " + " ".join(names.get(c, c) for c in mem[:14]))
