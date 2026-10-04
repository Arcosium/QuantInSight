"""Fundamental research rigor helpers inspired by ai-berkshire.

This module is intentionally deterministic and side-effect free. It does not
call an LLM or a broker; callers pass DART/disclosure text and any numeric
facts they already have. The result is an advisory research snapshot that can
be stored, attributed, and injected into agent prompts without changing order
execution semantics.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional


HIGH_RISK_PATTERNS = (
    "관리종목", "거래정지", "상장폐지", "횡령", "배임", "감사의견 거절", "의견거절",
    "회계감리", "불성실공시", "자본잠식", "감자", "대규모 유상증자",
)
MEDIUM_RISK_PATTERNS = (
    "전환사채", "신주인수권", "CB", "BW", "유상증자", "소송", "제재", "영업손실",
    "당기순손실", "매출 감소", "조회 실패", "시스템 리스크",
)


def _decimal(v: Any) -> Optional[Decimal]:
    if v is None or v == "":
        return None
    if isinstance(v, Decimal):
        return v
    try:
        return Decimal(str(v).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def _pct_diff(a: Decimal, b: Decimal) -> Optional[float]:
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    return float(abs(a - b) / base * Decimal("100"))


def validate_market_cap(price: Any, shares_out: Any, reported_market_cap: Any,
                        tolerance_pct: float = 5.0) -> Dict[str, Any]:
    """Validate reported market cap against price * shares_out.

    Returns a structured check dict. Missing inputs are represented as SKIPPED,
    not failure, because ArQuant often has partial data in live cycles.
    """
    p = _decimal(price)
    s = _decimal(shares_out)
    r = _decimal(reported_market_cap)
    if p is None or s is None or r is None:
        return {"name": "market_cap", "state": "SKIPPED", "note": "시총 검산 입력 부족"}
    calc = p * s
    diff = _pct_diff(calc, r)
    state = "OK" if diff is not None and diff <= float(tolerance_pct) else "WARN"
    return {
        "name": "market_cap",
        "state": state,
        "computed": float(calc),
        "reported": float(r),
        "diff_pct": round(float(diff or 0.0), 3),
        "note": f"시총 검산 {'통과' if state == 'OK' else '불일치'}: 계산 {calc:,.0f}, 보고 {r:,.0f}, 차이 {float(diff or 0.0):.2f}%",
    }


def extract_risk_flags(text: str) -> Dict[str, List[str]]:
    """Extract high/medium disclosure risk flags from Korean DART text."""
    t = text or ""
    high = [p for p in HIGH_RISK_PATTERNS if p.lower() in t.lower()]
    medium = [p for p in MEDIUM_RISK_PATTERNS if p.lower() in t.lower()]
    return {"high": sorted(set(high)), "medium": sorted(set(medium))}


def _score_from_flags(high: Iterable[str], medium: Iterable[str], has_verified_bs: bool) -> Dict[str, float]:
    high_n = len(list(high))
    medium_n = len(list(medium))
    quality = 7.0 - high_n * 2.5 - medium_n * 0.6
    if has_verified_bs:
        quality += 0.7
    quality = max(0.0, min(10.0, quality))
    valuation = 5.0
    moat = max(3.0, min(7.0, quality - 0.5))
    management = max(2.0, min(8.0, quality - (1.0 if high_n else 0.0)))
    return {
        "business_quality_score": round(quality, 1),
        "valuation_margin_score": round(valuation, 1),
        "moat_score": round(moat, 1),
        "management_score": round(management, 1),
    }


def assess_fundamental_research(code: str, name: str = "", *,
                                dart_text: str = "", financial_text: str = "",
                                price: Any = None, shares_out: Any = None,
                                reported_market_cap: Any = None,
                                source: str = "dart") -> Dict[str, Any]:
    """Build an advisory fundamental research snapshot.

    Verdict semantics:
    - QUALITY_VETO: deterministic high-severity disclosure/accounting risk.
    - WATCH: non-fatal but material issue; agents should discuss it.
    - ADVISORY_ONLY: no hard fundamental issue found from available inputs.
    """
    text = "\n".join(x for x in (dart_text or "", financial_text or "") if x)
    flags = extract_risk_flags(text)
    has_verified_bs = "✅ 재무상태표 검증" in text
    checks = [validate_market_cap(price, shares_out, reported_market_cap)]
    if "⚠️ 재무상태표 내적 불일치" in text:
        checks.append({"name": "balance_sheet_sanity", "state": "WARN", "note": "DART 재무상태표 내적 불일치"})
    elif has_verified_bs:
        checks.append({"name": "balance_sheet_sanity", "state": "OK", "note": "DART 재무상태표 검증 통과"})
    else:
        checks.append({"name": "balance_sheet_sanity", "state": "SKIPPED", "note": "검증된 재무상태표 텍스트 없음"})

    scores = _score_from_flags(flags["high"], flags["medium"], has_verified_bs)
    invalidators: List[str] = []
    invalidators.extend(f"high:{x}" for x in flags["high"])
    invalidators.extend(f"medium:{x}" for x in flags["medium"])
    invalidators.extend(c["note"] for c in checks if c.get("state") == "WARN")

    if flags["high"] or any(c.get("name") == "balance_sheet_sanity" and c.get("state") == "WARN" for c in checks):
        verdict = "QUALITY_VETO"
    elif flags["medium"] or any(c.get("state") == "WARN" for c in checks):
        verdict = "WATCH"
    else:
        verdict = "ADVISORY_ONLY"

    memo_bits = []
    if flags["high"]:
        memo_bits.append("중대 공시/재무 리스크: " + ", ".join(flags["high"]))
    if flags["medium"]:
        memo_bits.append("주의 신호: " + ", ".join(flags["medium"]))
    if not memo_bits:
        memo_bits.append("가용 DART/재무 텍스트 기준 명시적 품질 veto 없음")
    memo_bits.extend(c["note"] for c in checks if c.get("state") != "SKIPPED")

    return {
        "code": str(code or "").strip(),
        "name": name or str(code or "").strip(),
        "source": source,
        "verdict": verdict,
        "thesis_invalidators": invalidators,
        "financial_checks": checks,
        "memo": " | ".join(memo_bits)[:2000],
        **scores,
    }


def format_research_for_prompt(research: Optional[Dict[str, Any]]) -> str:
    if not research:
        return ""
    inv = research.get("thesis_invalidators") or []
    return (
        f"[펀더멘털 리서치 참고] verdict={research.get('verdict')} | "
        f"품질 {research.get('business_quality_score')} / 밸류 {research.get('valuation_margin_score')} / "
        f"해자 {research.get('moat_score')} / 경영 {research.get('management_score')} | "
        f"thesis invalidators: {', '.join(inv) if inv else '없음'} | "
        f"{research.get('memo') or ''}"
    )

