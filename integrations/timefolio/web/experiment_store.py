"""브라우저 프로필별 전략 실험 기록과 전략 전환 처리."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

_LOCK = threading.RLock()
_PROFILES = config.PRIVATE_DATA_DIR / "profiles"
_ACTIVE = config.PRIVATE_DATA_DIR / "active_strategy.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _profile_key(profile_id: str | None) -> str:
    raw = (profile_id or "local-default").strip()[:200]
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def strategy_id(strategy: dict) -> str:
    body = json.dumps(strategy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _path(profile_id: str | None) -> Path:
    return _PROFILES / f"{_profile_key(profile_id)}.json"


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _duration(started_at: str, ended_at: str) -> tuple[float, float]:
    try:
        delta = datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        hours = max(0.0, delta.total_seconds() / 3600)
    except (TypeError, ValueError):
        hours = 0.0
    return round(hours, 2), round(hours / 24, 2)


def _active_metrics(active: dict, nav: float | None, signal_count: int) -> dict:
    current = float(nav or active.get("start_nav") or 0)
    start = float(active.get("start_nav") or current or 0)
    ret = ((current / start) - 1) * 100 if start > 0 else 0.0
    now = _now()
    hours, days = _duration(active.get("started_at", now), now)
    active.update(current_nav=current, return_pct=round(ret, 4),
                  signal_count=int(signal_count), elapsed_hours=hours, elapsed_days=days,
                  updated_at=now)
    return active


def get(profile_id: str | None, *, nav: float | None = None,
        signal_count: int = 0) -> dict:
    with _LOCK:
        path = _path(profile_id)
        data = _read(path, {"history": [], "backtests": []})
        if data.get("active"):
            data["active"] = _active_metrics(data["active"], nav, signal_count)
            _write(path, data)
        return data


def record_backtest(profile_id: str | None, strategy: dict, stats: dict) -> None:
    with _LOCK:
        path = _path(profile_id)
        data = _read(path, {"history": [], "backtests": []})
        data.setdefault("backtests", []).append({
            "strategy_id": strategy_id(strategy), "strategy": strategy,
            "stats": stats, "tested_at": _now(),
        })
        data["backtests"] = data["backtests"][-50:]
        _write(path, data)


def activate(profile_id: str | None, strategy: dict, *, nav: float | None,
             signal_count: int, confirm_reset: bool = False) -> dict:
    sid = strategy_id(strategy)
    with _LOCK:
        path = _path(profile_id)
        data = _read(path, {"history": [], "backtests": []})
        current = data.get("active")
        if current and current.get("strategy_id") != sid and not confirm_reset:
            current = _active_metrics(current, nav, signal_count)
            return {"requires_reset": True, "current": current,
                    "message": "전략을 바꾸면 지금까지의 전략 수익률과 매매 신호 로그가 초기화됩니다. 계속하시겠습니까?"}
        if current and current.get("strategy_id") == sid:
            return {"requires_reset": False, "active": current, "changed": False}
        now = _now()
        if current:
            current = _active_metrics(current, nav, signal_count)
            hours, days = _duration(current.get("started_at", now), now)
            current.update(ended_at=now, elapsed_hours=hours, elapsed_days=days,
                           status="switched")
            data.setdefault("history", []).append(current)
            data["history"] = data["history"][-100:]
        active = {
            "strategy_id": sid, "strategy": strategy, "started_at": now,
            "start_nav": float(nav or 0), "current_nav": float(nav or 0),
            "return_pct": 0.0, "signal_count": 0, "elapsed_hours": 0.0,
            "elapsed_days": 0.0, "status": "active", "updated_at": now,
        }
        data["active"] = active
        _write(path, data)
        _write(_ACTIVE, {"profile_key": _profile_key(profile_id), **active})
        return {"requires_reset": False, "active": active,
                "changed": True, "had_previous": bool(current)}


def active_global() -> dict | None:
    with _LOCK:
        return _read(_ACTIVE, None)


def mark_evaluated(date_text: str) -> None:
    with _LOCK:
        active = _read(_ACTIVE, None)
        if active:
            active["last_evaluated_date"] = date_text
            active["updated_at"] = _now()
            _write(_ACTIVE, active)


def reset_strategy_runtime(previous_strategy_id: str = "strategy") -> Path:
    """전략 모의장부만 보관 후 비운다. Auto-folio 장부는 대상이 아니다."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = config.PRIVATE_DATA_DIR / "strategy_archives" / f"{stamp}-{previous_strategy_id[:16]}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in (config.SIGNALS_JSONL, config.TRADES_CSV, config.POSITIONS_JSON):
        if path.exists() and path.is_file():
            shutil.copy2(path, archive / path.name)
    config.SIGNALS_JSONL.write_text("", encoding="utf-8")
    config.TRADES_CSV.write_text("", encoding="utf-8")
    config.POSITIONS_JSON.write_text("[]\n", encoding="utf-8")
    return archive


def signal_count() -> int:
    try:
        with config.SIGNALS_JSONL.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0
