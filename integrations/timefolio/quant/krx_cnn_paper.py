"""Forward-only local paper accounts. Never retroactively invent an entry."""
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from arcmarket.systematic import Book, features, select
from quant.krx_cnn_data import atomic_json, load


def eligible_execution(date, decision):
    created = pd.Timestamp(decision["created_at"]).tz_convert("Asia/Seoul")
    opening = pd.Timestamp(date).tz_localize("Asia/Seoul")+pd.Timedelta(hours=9)
    return created < opening and pd.Timestamp(decision["as_of"]) < pd.Timestamp(date)


def advance(state, histories, *, now=None):
    """A persisted decision is required before a session opens. Idempotent."""
    today = pd.Timestamp(now or pd.Timestamp.now(tz="Asia/Seoul").date()).tz_localize(None).normalize()
    last = pd.Timestamp(state["last_processed"])
    dates = sorted({d for h in histories.values() for d in h.index if last < d < today})
    for date in dates:
        bars = {c: h.loc[date].to_dict() for c, h in histories.items() if date in h.index}
        for name, account in state["accounts"].items():
            book = Book(**account["book"])
            pending = account.get("pending")
            weights = None
            if pending and eligible_execution(date, pending):
                # Orders expire when no following session was observed promptly.
                if (date-pd.Timestamp(pending["as_of"])).days <= 7:
                    weights = pending["weights"]
                account["pending"] = None
            nav = book.step(date, bars, weights=weights,
                            previous=(pending or {}).get("previous", {}), adv=(pending or {}).get("adv", {}))
            account["book"] = asdict(book)
            account["curve"].append({"d": str(date.date()), "v": nav/10_000_000.})
        state["last_processed"] = str(date.date())
    return state


def update(root, signal):
    root = Path(root); path = root/"paper.json"
    histories = load(); frame = features(histories)
    if frame.empty:
        return None
    asof = frame.date.max(); day = frame[frame.date == asof]
    state = json.loads(path.read_text()) if path.exists() else {
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(), "last_processed": str(asof.date()),
        "last_decision_asof": None, "mode": "local_forward_paper", "accounts": {
            k: {"book": asdict(Book()), "curve": [{"d": str(asof.date()), "v": 1.}], "pending": None}
            for k in ("cnn", "momentum", "equal_weight")}}
    original_id = state.get("model_id") or (state["accounts"]["cnn"].get("pending") or {}).get("model_id")
    if original_id and original_id != signal.get("model_id"):
        raise ValueError("model_changed_keep_existing_paper_history_separate")
    state["model_id"] = signal.get("model_id")
    advance(state, histories)
    last = state.get("last_decision_asof")
    age = len(frame.loc[frame.date > pd.Timestamp(last), "date"].unique()) if last else 5
    if age >= 5 and signal.get("status") == "ready":
        created = pd.Timestamp.now(tz="UTC").isoformat()
        previous = {str(r.code): float(r.close) for r in day.itertuples() if pd.notna(r.close)}
        adv = {str(r.code): float(r.adv) for r in day.itertuples() if pd.notna(r.adv)}
        plans = {"cnn": signal, "momentum": select(day), "equal_weight": select(day, method="equal_weight")}
        for name, plan in plans.items():
            state["accounts"][name]["pending"] = {"created_at": created, "as_of": str(asof.date()),
                "weights": plan["weights"], "model_id": signal["model_id"] if name=="cnn" else name,
                "previous": previous, "adv": adv}
        state["last_decision_asof"] = str(asof.date())
        atomic_json(root/"decisions"/(created.replace(":", "")+".json"),
                    {k: a["pending"] for k, a in state["accounts"].items()})
    atomic_json(path, state)
    return state
