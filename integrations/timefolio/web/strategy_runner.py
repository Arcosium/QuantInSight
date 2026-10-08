"""활성 AI 전략을 일별로 평가하고 Auto-folio가 읽는 신호로 변환한다."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import config
from utils import market_time as mt
from web import experiment_store as experiments

log = logging.getLogger("arctrade.ai_strategy")


def _append_recent(result: dict, active: dict) -> int:
    today = mt.now_kst().strftime("%Y-%m-%d")
    sid = active.get("strategy_id")
    existing = set()
    try:
        for line in config.SIGNALS_JSONL.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            existing.add((rec.get("strategy_id"), rec.get("kind"), rec.get("follower"), rec.get("signal_date")))
    except (OSError, ValueError):
        pass

    rows = []
    recent = result.get("recent_signals") or {}
    for side, kind in (("sell", "SELL"), ("buy", "BUY")):
        for item in recent.get(side) or []:
            if item.get("date") != today:
                continue
            code = str(item.get("code") or "").zfill(6)
            key = (sid, kind, code, today)
            if not code.strip("0") or key in existing:
                continue
            rows.append({
                "ts": mt.now_kst().isoformat(timespec="seconds"),
                "kind": kind,
                "follower": code,
                "price": item.get("price"),
                "reason": f"AI 전략 {active.get('strategy', {}).get('name') or sid}",
                "source": "ai_strategy",
                "strategy_id": sid,
                "signal_date": today,
            })
            existing.add(key)
    if rows:
        config.SIGNALS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with config.SIGNALS_JSONL.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def evaluate_active(force: bool = False, prepared_result: dict | None = None) -> dict:
    active = experiments.active_global()
    if not active:
        return {"ok": True, "skipped": "no_active_strategy"}
    today = mt.now_kst().strftime("%Y-%m-%d")
    if not force and active.get("last_evaluated_date") == today:
        return {"ok": True, "skipped": "already_evaluated"}
    result = prepared_result
    if result is None:
        from web import nlbacktest
        result = nlbacktest.run_strategy(active["strategy"], include_report=False, track_progress=False)
    if result.get("error"):
        return {"ok": False, "error": result["error"]}
    emitted = _append_recent(result, active)
    experiments.mark_evaluated(today)
    log.info("AI 전략 일별 평가 완료 strategy=%s emitted=%d", active.get("strategy_id"), emitted)
    return {"ok": True, "emitted": emitted, "strategy_id": active.get("strategy_id")}


async def auto_loop() -> None:
    """장중 활성 전략을 확인한다. 같은 거래일에는 한 번만 백테스트한다."""
    await asyncio.sleep(15)
    while True:
        try:
            if mt.in_order_session():
                await asyncio.to_thread(evaluate_active)
        except Exception:
            log.exception("AI 전략 자동 평가 실패")
        await asyncio.sleep(max(60, config.AI_STRATEGY_REFRESH_SEC))
