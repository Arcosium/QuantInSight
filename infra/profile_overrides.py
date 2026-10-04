"""
ArQuant — 프로필 한정 런타임 오버라이드 + 비관리자 제안 이력 (사장 피드백 2026-05-18)

배경:
  ArQuant 는 단일 프로세스·단일 소스다. ops_support_worker 가 .py 소스를 직접 고치고
  서버를 재시작하면 그 변경은 **모든 유저**에게 적용된다. 그래서:
    • ADMIN 계정(hh09080 / .env 시드)  → 기존대로 소스 수정+재시작 = 전체 반영
    • 비관리자 계정                     → 소스·서버 불가침. 대신 '튜닝 파라미터'만
                                          이 모듈을 통해 **해당 프로필 전용**으로 저장

격리 (Phase 1 → Phase 2):
  Phase 1 에선 credentials.set_active() 가 '활성 계정은 언제나 단 하나'를 강제해,
  활성 프로필 오버라이드를 runtime 전역 레이어에 올렸다가 전환 시 교체하는 방식으로
  격리를 흉내냈다. Phase 2 멀티테넌트로 전환하며 그 단일 활성 계정 개념(set_active)은
  폐지됐다. 프로필별 저장/조회(load/save/last_updated 등)는 그대로 uid 기준이라 정상이나,
  activate() 가 쓰던 runtime 전역 레이어 적용 경로는 더 이상 호출되지 않는다
  (유저별 runtime 레이어 분리는 별도 작업으로 남음 — Task 7 범위 밖 concern).

보안 경계:
  허용 키 = config.STRATEGY_TUNABLE_KEYS (UI '전략' 탭이 쓰는 그 화이트리스트).
  KIS 자격증명/계좌/소스 식별자는 구조적으로 이 목록에 없으므로 비관리자가
  param_overrides 로 자격증명을 밀어넣는 것은 불가능하다.

데이터 위치 (둘 다 .gitignore 대상 — data/ 하위):
  data/profiles/<uid>/overrides.json     {KEY: value, ...}  (튜닝 파라미터)
  data/profiles/<uid>/ops_history.json   [{...}]            (운용지원실장 제안 이력)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PROFILE_OV")
KST = timezone(timedelta(hours=9))

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_DIR = _DATA_DIR / "profiles"
_PROPOSAL_CAP = 500  # 프로필별 보관할 제안 이력 최대 건수


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _profile_dir(uid: int) -> Path:
    d = _PROFILES_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _overrides_path(uid: int) -> Path:
    return _profile_dir(uid) / "overrides.json"


def _history_path(uid: int) -> Path:
    return _profile_dir(uid) / "ops_history.json"


# ─── 화이트리스트 & 타입 강제 ──────────────────────────────────────────────────
def whitelist() -> set:
    """허용 가능한 오버라이드 키. config 의 큐레이션된 튜닝 키 목록을 그대로 재사용."""
    try:
        import config
        return set(config.STRATEGY_TUNABLE_KEYS)
    except Exception as e:
        logger.warning("whitelist 로드 실패: %s — 빈 집합(전부 거부)", e)
        return set()


def _coerce(key: str, value: Any) -> Any:
    """STRATEGY_KEY_META 기준으로 타입을 강제하고 min/max 로 클램프.
    LLM 이 문자열·범위 밖 값을 줘도 시스템이 깨지지 않도록 방어."""
    try:
        import config
        meta = (config.STRATEGY_KEY_META or {}).get(key, {})
    except Exception:
        meta = {}
    t = meta.get("type")
    try:
        if t == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
            return bool(value)
        if t == "int":
            v = int(float(value))
        elif t in ("pct_ratio", "pct_raw", "multiplier"):
            v = float(value)
        else:
            # 알 수 없는 타입 — 숫자면 float, 아니면 원본
            try:
                v = float(value)
            except (TypeError, ValueError):
                return value
        lo, hi = meta.get("min"), meta.get("max")
        # pct_ratio 는 '저장값=비율(0.10=10%)'이지만 LLM/사람은 % 숫자(10)를 준다.
        # 모호성 해소: v>1 이면 퍼센트로 보고 /100, v<=1 이면 이미 비율로 간주.
        # min/max(meta)는 UI 단위(예: 5~50)이므로 비율로 환산해 클램프.
        if t == "pct_ratio":
            if v > 1.0:
                v = v / 100.0
            if lo is not None:
                v = max(v, float(lo) / 100.0)
            if hi is not None:
                v = min(v, float(hi) / 100.0)
        else:
            if lo is not None:
                v = max(v, float(lo))
            if hi is not None:
                v = min(v, float(hi))
        return int(v) if t == "int" else v
    except (TypeError, ValueError) as e:
        logger.warning("_coerce 실패 key=%s value=%r: %s — 무시", key, value, e)
        return None


# ─── 오버라이드 저장/로드 ──────────────────────────────────────────────────────
def load(uid: Optional[int]) -> Dict[str, Any]:
    if uid is None:
        return {}
    p = _overrides_path(uid)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return {}
        wl = whitelist()
        # 디스크에 화이트리스트 밖 키가 있어도 (구버전·수기수정) 적용 단계에서 제외
        return {k: v for k, v in d.items() if k in wl}
    except Exception as e:
        logger.warning("overrides 로드 실패(uid=%s): %s", uid, e)
        return {}


def set_overrides(uid: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """params 중 화이트리스트 키만 타입강제·클램프 후 **기존값에 병합**하여 저장.
    반환: 실제로 반영된 {key: coerced_value}. (잘못된 키/값은 조용히 탈락)"""
    wl = whitelist()
    cur = load(uid)
    accepted: Dict[str, Any] = {}
    for k, raw in (params or {}).items():
        if k not in wl:
            continue
        cv = _coerce(k, raw)
        if cv is None:
            continue
        cur[k] = cv
        accepted[k] = cv
    if accepted:
        try:
            _overrides_path(uid).write_text(
                json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("overrides 저장 실패(uid=%s): %s", uid, e)
    return accepted


def last_updated(uid: Optional[int]) -> Optional[str]:
    """이 프로필 오버라이드가 마지막으로 갱신된 시각(KST 'YYYY-MM-DD HH:MM:SS').
    운용지원실장이 파라미터를 조정할 때마다 set_overrides 가 overrides.json 을 다시 쓰므로
    그 파일 mtime 이 곧 '마지막 파라미터 업데이트 시각'이다. 오버라이드가 없으면 None
    (→ 호출부가 전략 선택 시각 active.since 로 폴백). 사장 지시 2026-05-21."""
    if uid is None:
        return None
    p = _overrides_path(uid)
    if not p.exists():
        return None
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


# ─── 활성 프로필 → runtime 레이어 (LEGACY no-op) ───────────────────────────────
def activate(uid: Optional[int], is_admin: bool = False) -> Dict[str, Any]:
    """(LEGACY no-op) 과거: 활성 계정의 프로필 오버라이드를 runtime 전역 최상위 레이어로 올렸다.

    Task 10 멀티테넌트 마무리로 runtime.get(key, uid) 가 프로필 오버라이드를 디스크에서
    uid 별로 직접 읽도록 바뀌면서, 전역 레이어를 미리 올려둘 필요가 사라졌다(이 함수는
    이미 orphan 이었다). 하위호환을 위해 시그니처만 유지하고 적용 시점에 실제로 쓰일
    오버라이드를 (로깅/응답용으로) 반환만 한다 — runtime 전역 상태는 건드리지 않는다.

    반환: 적용 대상 오버라이드 dict (admin/uid=None 이면 빈 dict)."""
    ov = {} if (is_admin or uid is None) else load(uid)
    if ov:
        logger.info("프로필 오버라이드 (uid=%s keys=%s) — runtime.get(uid) 가 디스크에서 직접 적용", uid, list(ov.keys()))
    return ov


# ─── 비관리자 제안 이력 (프로필 전용) ──────────────────────────────────────────
def record_proposal(uid: int, meta: Dict[str, Any]) -> None:
    """비관리자 ops_support 워커가 제안한 내용을 프로필 전용 이력에 누적.
    소스는 건드리지 않았으므로 'applied' 가 아니라 'proposed'/'overrides_applied' 로 기록."""
    p = _history_path(uid)
    try:
        h = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        if not isinstance(h, list):
            h = []
    except Exception:
        h = []
    h.append({"ts": _now_kst(), **meta})
    if len(h) > _PROPOSAL_CAP:
        h = h[-_PROPOSAL_CAP:]
    try:
        p.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("record_proposal 저장 실패(uid=%s): %s", uid, e)


def load_proposals(uid: Optional[int]) -> List[Dict[str, Any]]:
    if uid is None:
        return []
    p = _history_path(uid)
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


