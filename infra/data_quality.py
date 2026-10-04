"""데이터 품질 모니터 (2026-06-15 ROI#5).

이번 세션 버그(당일봉 고정·부분체결 오기록·원장 괴리)의 재발을 막기 위한 자동 점검. 순수 함수
위주로, 매 사이클 보유 종목 일봉 CSV 이상과 KIS↔원장 수량 괴리를 표면화한다(silent corruption 방지).
"""
from __future__ import annotations
from datetime import datetime
from typing import List

import pandas as pd

_MAX_GAP_DAYS = 8        # 연속 거래일 간 달력 갭이 이보다 크면 의심(장기휴장 제외 위해 보수적)
_STALE_DAYS = 21         # 마지막 봉이 이보다 오래면 stale(수집 중단 의심)


def csv_issues(code: str, df: "pd.DataFrame", today_str: str) -> List[str]:
    """일봉 DataFrame 의 무결성 이슈 목록(빈 리스트=정상). 순수 함수."""
    issues: List[str] = []
    if df is None or len(df) == 0:
        return [f"{code}: 일봉 데이터 없음"]
    try:
        d = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        return [f"{code}: date 컬럼 파싱 불가"]
    # 중복 날짜
    dups = d[d.duplicated()].unique().tolist()
    if dups:
        issues.append(f"{code}: 중복 날짜 {dups[:3]}")
    # 말미 0거래량(당일 미완성봉이 정리 안 됨)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        if len(vol) and vol.iloc[-1] <= 0:
            issues.append(f"{code}: 말미 거래량 0 잔존({d.iloc[-1]}) — 당일 미완성봉 정리 누락 의심")
    # stale (마지막 봉이 너무 오래)
    try:
        last = datetime.strptime(d.iloc[-1], "%Y-%m-%d")
        today = datetime.strptime(today_str, "%Y-%m-%d")
        if (today - last).days > _STALE_DAYS:
            issues.append(f"{code}: 일봉 stale — 마지막 {d.iloc[-1]} ({(today-last).days}일 전, 수집 중단 의심)")
    except Exception:
        pass
    return issues


def ledger_drift_issues(reconcile_diffs: List[str]) -> List[str]:
    """trade_ledger.reconcile() 가 낸 KIS↔원장 수량 괴리 목록을 이슈로 변환(빈 입력=정상)."""
    return [f"원장 괴리 — {x}" for x in (reconcile_diffs or []) if x]


def persistent_drift_issues(diffs: List[str], streak_map: dict, *, threshold: int = 2) -> List[str]:
    """전이성(1회성·자가치유) 원장 괴리는 거르고, threshold 사이클 '연속' 지속된 것만 반환. (2026-06-22)

    부분체결 직후~5분 폴링 사이의 전이 괴리(KIS>원장, 다음 폴링이 잔량을 채움)가 매 사이클 푸시
    알림을 내던 노이즈를 차단한다. streak_map(종목코드→연속 횟수)을 in-place 갱신한다 —
    이번에 드리프트한 종목은 +1, 드리프트가 사라진 종목은 제거(리셋)한다. 입력 diffs 는
    reconcile() 의 원시 문자열('CODE: KIS N주 vs 원장 M주')이고, 반환도 같은 형식의 부분집합이다."""
    cur = {}
    for d in (diffs or []):
        tk = str(d).split(":")[0].strip()
        if tk:
            cur[tk] = d
    for tk in list(streak_map.keys()):
        if tk not in cur:
            del streak_map[tk]           # 괴리 해소 → 스트릭 리셋
    out = []
    for tk, d in cur.items():
        streak_map[tk] = int(streak_map.get(tk, 0)) + 1
        if streak_map[tk] >= int(threshold):
            out.append(d)
    return out


