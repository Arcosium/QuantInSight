"""Add historical admission rules before opening the sealed evaluation period.

The original CNN predictions remain valid: this amendment changes order admission,
not labels, images, optimisation or model weights. Re-score every validation
candidate under the amended rules before choosing the final model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from quant.timefolio_heatmap_data import ROOT, atomic_json
from quant.timefolio_heatmap_replay import replay
from quant.timefolio_heatmap_study import context, protocol, score_matrix, validation_score

SOURCES = {
    "administrative": ("MDCSTAT216_extended.json", "20220101", False),
    "investment_caution": ("MDCSTAT219_extended.json", "20220101", False),
    "attention": ("MDCSTAT230_corrected.json", "20240924", True),
    "warning": ("MDCSTAT233_extended.json", "20240101", False),
    "risk": ("MDCSTAT236_extended.json", "20220101", False),
}


def designation_mask(codes, dates, rows, *, single_session=False):
    """Designation is inclusive; release is effective from that date's opening."""
    index = {c: i for i, c in enumerate(codes)}
    dates = np.asarray(dates)
    banned = np.zeros((len(codes), len(dates)), bool)
    for row in rows:
        code = row["ISU_CD"]
        if code not in index:
            continue
        start = row["DESIGN_DD"].replace("/", "")
        if len(start) != 8:
            raise ValueError("Missing designation date")
        if single_session:
            active = dates == start
        else:
            end = row.get("RELEAS_DD", "-").replace("/", "")
            active = dates >= start
            if len(end) == 8:
                if end < start:
                    raise ValueError("Release precedes designation")
                active &= dates < end
        banned[index[code]] |= active
    return banned


def link_file(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def prepare(source, dest):
    source, dest = Path(source), Path(dest)
    if dest.exists():
        raise RuntimeError("Amended study directory already exists; preserve it.")
    guard = json.loads((source / "holdout.json").read_text())
    if guard.get("status") != "sealed_guard_not_evaluated":
        raise RuntimeError("Rule amendment must precede holdout evaluation.")
    p, ix, _, _ = context(source)
    blocked = np.zeros(p["eligible"].shape, bool)
    coverage, raw_files = {}, {}
    for kind, (filename, start, single) in SOURCES.items():
        path = source / "rule_history" / filename
        raw = path.read_bytes(); data = json.loads(raw); rows = data["output"]
        if not rows:
            raise ValueError(f"Empty official history for {kind}")
        mask = designation_mask(ix["codes"], ix["dates"], rows, single_session=single)
        blocked |= mask
        coverage[kind] = {"query_start": start, "query_end": "20260923", "records": len(rows),
                          "blocked_security_days": int(mask.sum()),
                          "eligible_signals_blocked_next_day": int((p["eligible"][:, :-1] & mask[:, 1:]).sum()),
                          "sha256": hashlib.sha256(raw).hexdigest()}
        raw_files[filename] = path
    p["trade_allowed"] = ~blocked
    dest.mkdir(parents=True)
    for filename in ["panel_index.json", "samples.npz", "timefolio_sectors.json", "compute_receipt.json", "price_basis_audit.json"]:
        link_file(source / filename, dest / filename)
    for file in source.glob("images_f*_w*.npy"):
        link_file(file, dest / file.name)
    for filename, path in raw_files.items():
        link_file(path, dest / "rule_history" / filename)
    np.savez_compressed(dest / "panel.npz", **p)
    audit = json.loads((source / "data_audit.json").read_text())
    audit["historical_designations"] = coverage
    audit["missing_rules"] = [x for x in audit["missing_rules"] if not x.startswith("historical warning")]
    audit["missing_rules"].append("designation query coverage is not exhaustively cross-checked against every KIND notice")
    audit["blocked_eligible_orders_next_day"] = int((p["eligible"][:, :-1] & blocked[:, 1:]).sum())
    atomic_json(dest / "data_audit.json", audit)
    atomic_json(dest / "rule_amendment.json", {
        "source_run": str(source), "holdout_opened": False,
        "reason": "Official historical designations recovered during CNN validation screening.",
        "unchanged": ["feature tensors", "training labels", "date partitions", "36 CNN configurations", "seeds", "selection formula", "9 GBM configurations"],
        "changes": ["block buys on designation-effective date", "repair individual stock weight drift", "protect other stock weight caps against fee-induced NAV shrinkage"],
        "training_universe": "Original liquid-stock training samples are retained; legal purchase eligibility is applied at execution only.",
        "intervals": "Attention applies on its designated session. Other designations apply from designation date inclusive to release date exclusive; unreleased records remain blocked.",
        "coverage": coverage,
        "attention_primary_source": "https://kind.krx.co.kr/external/2026/07/13/000543/20260713001222/70820.htm"})
    protocol(dest)
    print(json.dumps({"prepared": str(dest), "coverage": coverage,
                      "blocked_eligible_orders_next_day": audit["blocked_eligible_orders_next_day"]}), flush=True)


def rescore(source, dest):
    from quant.timefolio_heatmap_audit import audit_fills
    source, dest = Path(source), Path(dest)
    old = json.loads((source / "screen_summary.json").read_text())
    p, ix, ci, di = context(dest)
    reference = replay(p, ix, p["r5"], "20260401", "20260630")
    atomic_json(dest / "screen" / "momentum_reference.json", reference)
    updated = []
    for item in old:
        cfg = item["config"]; name = cfg["id"]
        for suffix in [".pred.npy", ".pt"]:
            link_file(source / "screen" / (name + suffix), dest / "screen" / (name + suffix))
        pred = np.load(source / "screen" / (name + ".pred.npy"))
        bt = replay(p, ix, score_matrix(pred, ci, di, p["close"].shape), "20260401", "20260630", rebalance=cfg["rebalance"], return_trades=True)
        row = copy.deepcopy(item)
        row["before_rule_amendment"] = {"portfolio": row["portfolio"], "selection_score": row["selection_score"]}
        row["portfolio"], row["selection_score"] = bt["metrics"], validation_score(bt, reference)
        row["independent_weight_audit"] = audit_fills(p, ix, bt)
        atomic_json(dest / "screen" / (name + ".json"), row)
        updated.append(row)
    updated.sort(key=lambda x: x["selection_score"], reverse=True)
    atomic_json(dest / "screen_summary.json", updated)
    # Existing three-seed ensembles can also be re-scored without any retraining.
    for item in updated[:3]:
        name = item["config"]["id"]
        path = source / "refine" / (name + ".json")
        if not path.exists():
            continue
        row = json.loads(path.read_text()); cfg = row["config"]
        pred = np.load(source / "refine" / (name + ".pred.npy"))
        bt = replay(p, ix, score_matrix(pred, ci, di, p["close"].shape), "20260401", "20260630", rebalance=cfg["rebalance"])
        row["before_rule_amendment"] = {"portfolio": row["portfolio"], "selection_score": row["selection_score"], "seed_metrics": row.pop("seed_metrics")}
        row["portfolio"], row["selection_score"] = bt["metrics"], validation_score(bt, reference)
        atomic_json(dest / "refine" / (name + ".json"), row)
        for file in (source / "refine").glob(name + "*"):
            if file.suffix != ".json": link_file(file, dest / "refine" / file.name)
    print(json.dumps({"rescored": len(updated), "top_three": [{"id": r["config"]["id"], "score": r["selection_score"]} for r in updated[:3]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["prepare", "rescore"])
    ap.add_argument("--source", type=Path, default=ROOT)
    ap.add_argument("--dest", type=Path, default=ROOT.with_name(ROOT.name + "_rules_v2"))
    a = ap.parse_args(); (prepare if a.command == "prepare" else rescore)(a.source, a.dest)
