"""Keep strategy changes separate from the strategy currently collecting evidence.

An LLM's proposal and a backtest of the current strategy are not validation of
the proposed strategy. Store exact baseline/proposal pairs for later comparison.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from infra import user_paths


def record_proposal(uid, overrides, *, rationale="", trigger="cycle"):
    import config
    import runtime
    from infra.ops_param_clamp import clamp_overrides
    candidate, _ = clamp_overrides(overrides)
    baseline = {k: runtime.get(k, uid=uid) for k in config.STRATEGY_TUNABLE_KEYS}
    payload = {"baseline": baseline, "overrides": candidate}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()[:20]
    folder = user_paths.user_dir(uid) / "research"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.json"
    record = dict(payload, id=digest, created_at=datetime.now(timezone.utc).isoformat(),
                  trigger=trigger, rationale=rationale, status="awaiting_validation",
                  requirements=["same-data baseline comparison after costs",
                                "chronological out-of-sample comparison",
                                "forward paper results with frozen parameters"],
                  applied=False)
    # Immutable, deduplicated snapshot. No later proposal rewrites its baseline.
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
    except FileExistsError:
        pass
    return digest


def list_proposals(uid, limit=30):
    folder = user_paths.user_dir(uid) / "research"
    records = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return records
