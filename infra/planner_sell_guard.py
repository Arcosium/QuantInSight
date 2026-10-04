"""포트폴리오기획팀장 매도 반론과 1회 보류권.

계획 보유기간이 남았고 목표가·손절가·thesis invalidator가 모두 없는 포지션은
첫 재량 매도를 한 사이클 보류한다. 다음 정기 사이클에서도 매도 판단이 반복되면 허용한다.
손절·목표 도달·중대 thesis 훼손은 보류하지 않는다.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Tuple

from infra import user_paths

_HOLD_WORDS = {"보유", "유지", "hold", "keep", "관망"}
_DEFERRAL_STALE_HOURS = 7 * 24.0


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_dt(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).split("+")[0].strip())
    except (TypeError, ValueError):
        return None


def assess_objection(thesis: Optional[Dict[str, Any]], holding: Dict[str, Any], *,
                      hold_days: Optional[float] = None,
                      invalidators: Optional[Iterable[Any]] = None,
                      now: Optional[datetime] = None) -> Dict[str, Any]:
    """저장 thesis와 권위 현재가로 매도 반론 상태를 결정한다."""
    thesis = dict(thesis or {})
    now = now or datetime.now()
    code = str(holding.get("code") or "").strip()
    name = str(holding.get("name") or code)
    cur = _f(holding.get("cur_price"))
    entry = _f(thesis.get("entry_price"))
    target = _f(thesis.get("target_price"))
    stop = _f(thesis.get("stop_price"))
    planned_h = _f(thesis.get("planned_hold_hours"))
    entry_at = _parse_dt(thesis.get("entry_ts"))
    invalidators = [str(x).strip() for x in (invalidators or []) if str(x).strip()]

    if not thesis or not entry_at or planned_h <= 0 or cur <= 0:
        return {"active": False, "hard_override": False, "code": code,
                "message": f"{name}({code}) 저장된 계획 정보가 불완전해 보류권을 적용하지 않습니다."}
    if stop > 0 and cur <= stop:
        return {"active": False, "hard_override": True, "code": code,
                "message": f"{name}({code}) 현재가 {cur:,.2f}가 손절가 {stop:,.2f} 이하라 즉시 매도를 허용합니다."}
    if target > 0 and cur >= target:
        return {"active": False, "hard_override": True, "code": code,
                "message": f"{name}({code}) 현재가 {cur:,.2f}가 목표가 {target:,.2f} 이상이라 계획에 따른 매도를 허용합니다."}
    if invalidators:
        return {"active": False, "hard_override": True, "code": code,
                "message": f"{name}({code}) thesis 훼손 신호({'; '.join(invalidators[:3])})가 있어 매도를 허용합니다."}

    planned_end = entry_at + timedelta(hours=planned_h)
    remaining_h = (planned_end - now).total_seconds() / 3600.0
    if remaining_h <= 0:
        return {"active": False, "hard_override": False, "code": code,
                "message": f"{name}({code}) 계획 보유기간이 끝나 보류권을 적용하지 않습니다."}

    held_txt = f"{float(hold_days):.1f}일 보유" if hold_days is not None else "보유기간 확인 중"
    price_plan = (f"현재가 {cur:,.2f}는 손절가 {stop:,.2f}와 목표가 {target:,.2f} 사이이고"
                  if stop > 0 and target > 0 else
                  f"현재가 {cur:,.2f}에서 계획기간 중 하드 청산 조건이 확인되지 않았고")
    return {
        "active": True, "hard_override": False, "code": code,
        "remaining_hours": round(remaining_h, 1),
        "message": (f"강한 매도 반대: {name}({code})는 {held_txt}이며 계획 종료까지 약 {remaining_h:.1f}시간 남았습니다. "
                    f"{price_plan} 확인된 thesis 훼손도 없습니다. "
                    "미세한 신호 변화만으로 청산하지 말고 계획을 유지하십시오. 매도를 고수하려면 다음 정기 사이클에서도 "
                    "추세 붕괴·공시 악재·수급 반전 같은 새 반증을 다시 제시해야 합니다."),
    }


def _load(uid: int) -> Dict[str, Dict[str, Any]]:
    path = user_paths.planner_sell_deferral_path(int(uid))
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(uid: int, data: Dict[str, Dict[str, Any]]) -> None:
    path = user_paths.planner_sell_deferral_path(int(uid))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def apply_one_cycle_deferral(uid: int, code: str, directive: str,
                             objection: Dict[str, Any], *,
                             now: Optional[datetime] = None,
                             cycle_key: str = "") -> Tuple[str, str]:
    """첫 재량 매도는 보유로 바꾸고, 다음 정기 사이클의 반복 매도는 허용한다.

    반환 ``(최종 지시, 상태)``. 상태는 deferred|released|bypass|cleared 중 하나다.
    """
    now = now or datetime.now()
    code = str(code or "").strip()
    directive = str(directive or "보유").strip()
    data = _load(int(uid))
    if directive.lower() in _HOLD_WORDS:
        if data.pop(code, None) is not None:
            _save(int(uid), data)
        return directive, "cleared"
    if not objection.get("active") or objection.get("hard_override"):
        if data.pop(code, None) is not None:
            _save(int(uid), data)
        return directive, "bypass"

    previous = data.get(code) or {}
    previous_at = _parse_dt(previous.get("ts"))
    fresh = previous_at is not None and (now - previous_at).total_seconds() <= _DEFERRAL_STALE_HOURS * 3600
    previous_cycle = str(previous.get("cycle_key") or "")
    if fresh and (not cycle_key or not previous_cycle or cycle_key != previous_cycle):
        data.pop(code, None)
        _save(int(uid), data)
        return directive, "released"
    if fresh and cycle_key and cycle_key == previous_cycle:
        return "보유", "deferred"

    data[code] = {"ts": now.isoformat(timespec="seconds"), "directive": directive,
                  "cycle_key": str(cycle_key or ""),
                  "reason": str(objection.get("message") or "")[:500]}
    _save(int(uid), data)
    return "보유", "deferred"
