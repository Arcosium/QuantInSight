"""섀도우 프로파일 비교 — 저비용 알파풀 (P3, 사장 지시 2026-08-02).

배경(Medium 'Managing Real-Life Portfolio based on Multi-Agent LLMs' 이식):
  단일 최적점으로 수렴시키는 대신 **상관 낮은 여러 전략을 동시에 굴려 커버리지를 넓히는 것**이
  요지다. 실계좌는 하나뿐이라 알파 N개를 동시에 태울 수 없지만, QIS 는 이미 KIS 모의 계정 uid
  여러 개가 각자 profile_overrides 로 **서로 다른 파라미터**를 들고 병렬로 스웜을 돌린다.
  그 계정들이 곧 공짜 섀도우 알파풀이다 — 여기서는 그 계정들의 자산곡선을 읽어
  성과(샤프)와 **프로필 간 상관**을 계산하고, 실계좌 대비 '더 낫고 덜 닮은' 프로필을 지목한다.

정직성:
  • 모든 값은 입출금 보정(total_eval − external_flow_cum) 후 산출한다. 보정 안 하면 입금이
    수익으로 잡힌다.
  • 관측은 사이클/폴 단위(하루 여러 건)이므로 **날짜별 마지막 관측**으로 일간화한다.
  • sharpe_like = mean/std×√252 (무위험수익률 미차감) — 이름 그대로 근사치다.
  • 상관은 두 프로필이 **공통으로 관측된 날**만 쓴다. 공통일이 부족하면 None(무음 금지).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SHADOW")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MIN_DAYS = 5             # 이보다 짧으면 통계를 내지 않는다
_MIN_COMMON_DAYS = 5      # 상관 계산 최소 공통일
_LOW_CORR = 0.7           # 이 미만이면 '분산 효과 있음'으로 본다


def _uid_dirs() -> List[int]:
    """data/<uid>/equity_curve.json 을 가진 계정 uid 목록 (백업 폴더 등 비숫자 제외)."""
    out = []
    for p in (PROJECT_ROOT / "data").glob("*/equity_curve.json"):
        name = p.parent.name
        if name.isdigit():
            out.append(int(name))
    return sorted(out)


def daily_equity(uid: int) -> Dict[str, float]:
    """{'YYYY-MM-DD': 입출금보정 평가액} — 날짜별 마지막 관측."""
    p = PROJECT_ROOT / "data" / str(int(uid)) / "equity_curve.json"
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("equity_curve 로드 실패(uid=%s): %s", uid, e)
        return {}
    out: Dict[str, float] = {}
    for r in rows if isinstance(rows, list) else []:
        try:
            ts, ev = str(r.get("ts") or ""), r.get("total_eval")
            if len(ts) < 10 or not ev:
                continue
            out[ts[:10]] = float(ev) - float(r.get("external_flow_cum") or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def _returns(series: Dict[str, float]) -> Dict[str, float]:
    """{날짜: 평가액} → {날짜: 전일대비 수익률}. 값이 0/음수인 날은 건너뛴다(결제 글리치 가드)."""
    dates = sorted(series)
    out: Dict[str, float] = {}
    for prev, cur in zip(dates, dates[1:]):
        a, b = series[prev], series[cur]
        if a > 0 and b > 0:
            out[cur] = b / a - 1.0
    return out


def _stats(series: Dict[str, float]) -> Dict[str, Any]:
    """구간 수익률·샤프근사·MDD."""
    dates = sorted(series)
    rets = list(_returns(series).values())
    n = len(rets)
    first, last = series[dates[0]], series[dates[-1]]
    peak, mdd = first, 0.0
    for d in dates:
        peak = max(peak, series[d])
        if peak > 0:
            mdd = min(mdd, series[d] / peak - 1.0)
    sharpe = None
    if n >= 2:
        mean = sum(rets) / n
        std = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
        sharpe = round(mean / std * (252 ** 0.5), 2) if std > 1e-12 else None
    return {"days": len(dates), "period": f"{dates[0]} ~ {dates[-1]}",
            "return_pct": round((last / first - 1.0) * 100, 2) if first > 0 else None,
            "sharpe_like": sharpe, "mdd_pct": round(mdd * 100, 2)}


def compare(uids: Optional[List[int]] = None, *, base_uid=None,
            min_days: int = _MIN_DAYS) -> Dict[str, Any]:
    """프로필별 성과 + 프로필 간 상관 + base_uid 관점의 분산 후보.

    반환 {available, profiles{uid:...}, correlation{'a|b':r}, diversifiers[str], note}."""
    from tools.agent_scorecard import pearson
    series = {u: daily_equity(u) for u in (uids if uids is not None else _uid_dirs())}
    series = {u: s for u, s in series.items() if len(s) >= min_days}
    if len(series) < 2:
        return {"available": False,
                "reason": f"비교 가능한 프로필 부족({len(series)}개, {min_days}일 이상 필요)"}

    profiles: Dict[str, Any] = {}
    for u, s in series.items():
        st = _stats(s)
        try:
            from infra import profile_overrides
            st["overrides"] = profile_overrides.load(u) or {}
        except Exception:
            st["overrides"] = {}
        profiles[str(u)] = st

    rets = {u: _returns(s) for u, s in series.items()}
    corr: Dict[str, Optional[float]] = {}
    corr_days: Dict[str, int] = {}
    us = sorted(series)
    for i, a in enumerate(us):
        for b in us[i + 1:]:
            common = sorted(set(rets[a]) & set(rets[b]))
            corr_days[f"{a}|{b}"] = len(common)
            r = (pearson([rets[a][d] for d in common], [rets[b][d] for d in common])
                 if len(common) >= _MIN_COMMON_DAYS else None)
            corr[f"{a}|{b}"] = round(r, 3) if r is not None else None

    return {"available": True, "profiles": profiles, "correlation": corr,
            "corr_days": corr_days,
            "diversifiers": _diversifiers(base_uid, profiles, corr),
            "note": "sharpe_like=평균/표준편차×√252(무위험 미차감). 상관은 공통 관측일의 일간수익률 "
                    f"피어슨 — |r|<{_LOW_CORR} 면 분산 효과가 있다고 본다. "
                    "⚠️ 표본이 수십일 미만이면 샤프·상관 모두 잡음이 크다(corr_days 확인). "
                    "⚠️ KIS 모의계정은 해외 시세·평단이 결손일 수 있어(trade_ledger 참조) "
                    "US 비중이 큰 모의 프로필의 수익률은 액면 그대로 믿지 말 것."}


def _corr_between(corr: Dict[str, Optional[float]], a, b) -> Optional[float]:
    return corr.get(f"{a}|{b}", corr.get(f"{b}|{a}"))


def _param_diff(base: Dict[str, Any], other: Dict[str, Any], limit: int = 6) -> str:
    """두 프로필 오버라이드의 차이 — '무엇이 성과 차이를 만들었나'의 후보."""
    keys = sorted(set(base) | set(other))
    diffs = [f"{k} {base.get(k, '기본')}→{other.get(k, '기본')}" for k in keys
             if base.get(k) != other.get(k)]
    return ", ".join(diffs[:limit]) + (f" 외 {len(diffs) - limit}건" if len(diffs) > limit else "")


def _diversifiers(base_uid, profiles: Dict[str, Any], corr: Dict[str, Optional[float]]) -> List[str]:
    """base 대비 **샤프가 높으면서 상관이 낮은** 프로필 — 채택 검토 1순위."""
    if base_uid is None or str(base_uid) not in profiles:
        return []
    base = profiles[str(base_uid)]
    bs = base.get("sharpe_like")
    out = []
    for u, st in profiles.items():
        if u == str(base_uid):
            continue
        r, s = _corr_between(corr, int(base_uid), int(u)), st.get("sharpe_like")
        if s is None or bs is None or s <= bs:
            continue
        rr = "상관 미상(공통일 부족)" if r is None else f"상관 {r:+.2f}"
        mark = " ← 저상관·고성과" if (r is not None and abs(r) < _LOW_CORR) else ""
        diff = _param_diff(base.get("overrides") or {}, st.get("overrides") or {})
        out.append(f"uid {u}: 샤프 {s} (내 {bs}) · {rr} · 수익률 {st.get('return_pct')}%{mark}"
                   + (f" · 파라미터 차이: {diff}" if diff else " · 파라미터 차이 없음"))
    return out


def summary_lines(result: Dict[str, Any]) -> List[str]:
    if not result.get("available"):
        return [f"• 섀도우 프로파일 비교: 불가({result.get('reason', '사유 미상')})"]
    lines = ["• 프로파일별 성과: " + " / ".join(
        f"uid {u} 샤프 {st.get('sharpe_like')} 수익 {st.get('return_pct')}% MDD {st.get('mdd_pct')}%"
        for u, st in (result.get("profiles") or {}).items())]
    days = result.get("corr_days") or {}
    pairs = [(k, v) for k, v in (result.get("correlation") or {}).items() if v is not None]
    if pairs:
        lines.append("• 프로파일 간 상관: "
                     + ", ".join(f"{k} {v:+.2f}({days.get(k, '?')}일)" for k, v in pairs))
    for d in (result.get("diversifiers") or []):
        lines.append(f"  ↗ 채택 검토 — {d}")
    return lines
