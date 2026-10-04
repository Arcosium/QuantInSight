"""정책 플래그 변경 승인 인박스 (per-uid). 토요일 ops 제안 → 사장 승인 시 오버라이드 적용.

Coresight 인박스(infra/coresight_inbox)와 같은 pending 패턴이되, 승인 시 지시문이 아니라
profile_overrides.set_overrides 로 실제 플래그를 적용한다. 저장: data/profiles/<uid>/policy_pending.json.
거버넌스 2026-06-05: 평일 ops 는 정책 키 차단, 토요일(weekly)만 이 인박스로 회부한다."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("POLICY_APPROVAL_INBOX")
KST = timezone(timedelta(hours=9))
_PROFILES_DIR = Path(__file__).parent.parent / "data" / "profiles"
_FILENAME = "policy_pending.json"


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _path(uid: int) -> Path:
    d = _PROFILES_DIR / str(int(uid))
    d.mkdir(parents=True, exist_ok=True)
    return d / _FILENAME


def _load(uid: int) -> List[Dict[str, Any]]:
    p = _path(uid)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("policy_approval_inbox 로드 실패(uid=%s): %s", uid, e)
        return []


def _save(uid: int, items: List[Dict[str, Any]]) -> None:
    try:
        _path(uid).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("policy_approval_inbox 저장 실패(uid=%s): %s", uid, e)


def enqueue(uid: int, key: str, proposed_value: Any, current_value: Any,
            rationale: str = "") -> Optional[Dict[str, Any]]:
    """정책 키 변경 제안을 승인 대기함에 적재. 같은 key 가 있으면 최신값으로 갱신(pending 으로 리셋).

    사장 지시 2026-07-21: proposed == current(예: true→true) 는 실제 변경이 아니므로 적재하지 않고,
    이미 쌓여 있던 같은 key 의 no-op pending 항목도 제거한다(혼란스러운 무변경 승인 요청 정리)."""
    items = _load(uid)
    if proposed_value == current_value:
        pruned = [i for i in items if i.get("key") != key]
        if len(pruned) != len(items):
            _save(uid, pruned)
        return None
    item = {
        "id": key, "key": key, "proposed_value": proposed_value,
        "current_value": current_value, "rationale": rationale or "",
        "proposed_at": _now_kst(), "status": "pending",
        "label": "정책 변경 승인 대기(토요일 점검)",
    }
    items = [i for i in items if i.get("key") != key]
    items.append(item)
    _save(uid, items)
    logger.info("정책 변경 제안 적재 (uid=%s key=%s → %r)", uid, key, proposed_value)
    return item


def list_pending(uid: Optional[int]) -> List[Dict[str, Any]]:
    if uid is None:
        return []
    # no-op(proposed == current) 항목은 표시하지 않는다 — 과거에 쌓인 true→true 류 정리(사장 지시 2026-07-21).
    return list(reversed([i for i in _load(uid)
                          if i.get("status") == "pending"
                          and i.get("proposed_value") != i.get("current_value")]))


def approve(uid: int, key: str) -> bool:
    """대기 항목을 승인 → profile_overrides.set_overrides 로 적용, status=approved."""
    items = _load(uid)
    target = next((i for i in items if i.get("key") == key and i.get("status") == "pending"), None)
    if target is None:
        return False
    try:
        from infra import profile_overrides
        profile_overrides.set_overrides(int(uid), {key: target["proposed_value"]})
    except Exception as e:
        logger.warning("정책 승인 적용 실패(uid=%s key=%s): %s", uid, key, e)
        return False
    target["status"] = "approved"
    target["approved_at"] = _now_kst()
    _save(uid, items)
    logger.info("정책 변경 승인·적용 (uid=%s key=%s)", uid, key)
    return True


def reject(uid: int, key: str) -> bool:
    """대기 항목을 거부 → 큐에서 제거(적용 안 함)."""
    items = _load(uid)
    new_items = [i for i in items if not (i.get("key") == key and i.get("status") == "pending")]
    if len(new_items) == len(items):
        return False
    _save(uid, new_items)
    logger.info("정책 변경 거부 (uid=%s key=%s)", uid, key)
    return True
