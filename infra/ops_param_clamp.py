"""운용지원실장 파라미터 오버라이드 가드레일 — 적용 직전 범위 클램프/타입 정규화.

반려가 아니라 보정(HARD_MAX_ORDER_QTY 패턴과 동일 철학). 규칙:
  - 튜넌 화이트리스트(STRATEGY_TUNABLE_KEYS) 밖 키 → 드롭(+note).
  - 튜넌이지만 META 없음 → 통과(범위 정보 없어 클램프 불가, 합법 키이므로 드롭 금지).
  - bool → 정규화. choice → choices 검증(밖이면 드롭). 수치형 → [min,max] 클램프(int 라운드).
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y", "t")


def partition_protected(overrides: Dict[str, Any],
                        trigger: str = "cycle") -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """트리거별로 정책 플래그(config.OPS_PROTECTED_KEYS)를 분류. 반환 (kept, to_review, notes).

    회사 운영 거버넌스(사장 지시 2026-06-05): ops 는 전술 파라미터만 자율 조정한다.
    정책 플래그(자산군·엔진)는 —
      • manual(사장 직접지시) → 전부 kept(권위, 즉시 적용)
      • weekly(토요일 점검)   → 정책 키는 to_review(사장 승인 대기로 회부), 나머지 kept
      • cycle(평일·시간당)    → 정책 키 드롭(note), 나머지 kept"""
    import config
    protected = set(getattr(config, "OPS_PROTECTED_KEYS", ()))
    src = dict(overrides or {})
    if trigger == "manual":
        return src, {}, []
    kept: Dict[str, Any] = {}
    review: Dict[str, Any] = {}
    notes: List[str] = []
    for k, v in src.items():
        if k not in protected:
            kept[k] = v
        elif trigger == "weekly":
            review[k] = v
            notes.append(f"{k}: 정책 키 — 토요일 점검 → 사장 승인 대기로 회부")
        else:  # cycle
            notes.append(f"{k}: 정책 키 — 운용지원 자율 변경 불가(사장 전용) → 무시")
    return kept, review, notes


def partition_by_tier(overrides: Dict[str, Any],
                      trigger: str = "cycle") -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """파라미터를 cycle/weekly tier 로 분류해 트리거별 적용 가능분을 가른다 (사장 지시 2026-06-09 #7).
    반환 (apply, defer, notes).

    각 키의 tier 는 config.STRATEGY_KEY_META[k]['tier'] (없으면 'cycle').
      • cycle(평일·시간당) → weekly-tier 키는 defer(토요일 검증 큐로 회부), cycle-tier 만 apply.
      • weekly(토요일)     → 백테스트가 돈 컨텍스트이므로 전부 apply.
      • manual(사장 직접)  → 전부 apply(권위).
    weekly-tier = 점수엔진 가중치·사이징 모델·유니버스·종목수·슬리브 구조값 등(모델·구조 변경)."""
    import config
    meta = getattr(config, "STRATEGY_KEY_META", {})
    src = dict(overrides or {})
    if trigger in ("weekly", "manual"):
        return src, {}, []
    apply: Dict[str, Any] = {}
    defer: Dict[str, Any] = {}
    notes: List[str] = []
    for k, v in src.items():
        tier = (meta.get(k) or {}).get("tier", "cycle")
        if tier == "weekly":
            defer[k] = v
            notes.append(f"{k}: 토요일 백테스트 검증 후 조정 키 — 사이클 적용 보류, 주간 검토 큐로 회부")
        else:
            apply[k] = v
    return apply, defer, notes


def clamp_overrides(overrides: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """STRATEGY_KEY_META 기준으로 overrides 를 보정. 반환 (clamped, notes)."""
    import config
    meta = getattr(config, "STRATEGY_KEY_META", {})
    tunable = set(getattr(config, "STRATEGY_TUNABLE_KEYS", list(meta.keys())))
    out: Dict[str, Any] = {}
    notes: List[str] = []
    for k, v in (overrides or {}).items():
        if k not in tunable:
            notes.append(f"{k}: 튜닝 허용 키가 아님 → 무시")
            continue
        m = meta.get(k)
        if not m:
            out[k] = v          # 튜넌이지만 META 없음 → 통과(클램프 불가, 드롭 금지)
            continue
        typ = m.get("type")
        if typ == "bool":
            out[k] = _to_bool(v)
            continue
        if typ == "choice":
            choices = m.get("choices") or []
            if v in choices:
                out[k] = v
            else:
                notes.append(f"{k}: 허용값 아님({v!r}, 허용 {choices}) → 무시")
            continue
        # 수치형(int/pct_raw/float 등)
        try:
            num = float(v)
        except Exception:
            notes.append(f"{k}: 숫자 아님({v!r}) → 무시")
            continue
        if not math.isfinite(num):
            notes.append(f"{k}: 유한한 숫자가 아님 → 무시")
            continue
        lo = m.get("min"); hi = m.get("max")
        orig = num
        if lo is not None and num < lo:
            num = lo
        if hi is not None and num > hi:
            num = hi
        if typ == "int":
            num = int(round(num))
        if num != orig:
            notes.append(f"{k}: {orig} → {num} (범위 [{lo},{hi}] 클램프)")
        out[k] = num
    return out, notes
