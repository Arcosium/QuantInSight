"""Preregistered screening, seed replication, and one sealed holdout evaluation.

Run remotely with only exported market tensors and this package. No credentials,
account snapshots, source databases, or broker modules are required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

from quant.timefolio_heatmap_data import ROOT, atomic_json
from quant.timefolio_heatmap_features import labels
from quant.timefolio_heatmap_replay import replay

VAL_START, TEST_START, END = "20260401", "20260701", "20260923"
BASE = dict(frequency=30, window=5, horizon=5, target="relative", info="full", layout="standard",
            width=8, kernel=[3, 3], lr=.001, dropout=.1, epochs=8, rebalance=5)


def configurations():
    cases = []
    def add(name, **kw): cases.append(dict(BASE, id=name, **kw))
    # One-factor changes isolate information, geometry, and optimisation effects.
    add("base")
    for x in ["price_volume", "no_rules", "summary_only", "no_context"]:
        add("info_" + x, info=x)
    for x in ["rules_top", "rules_middle", "interleave", "transpose", "rows_shuffled", "time_shuffled"]:
        add("layout_" + x, layout=x)
    for w in [3, 10]: add(f"window_{w}", window=w)
    for w in [3, 5, 10]: add(f"15m_window_{w}", frequency=15, window=w)
    for h in [3, 10]: add(f"horizon_{h}", horizon=h)
    add("absolute_target", target="absolute")
    add("width_16", width=16)
    add("width_32", width=32)
    add("kernel_5x3", kernel=[5, 3])
    add("kernel_3x5", kernel=[3, 5])
    add("lr_0003", lr=.0003)
    add("dropout_03", dropout=.3)
    add("rebalance_3", rebalance=3)
    # Limited predefined interactions, not an unrestricted Cartesian search.
    for w in [3, 10]:
        for lay in ["rules_top", "interleave"]:
            add(f"w{w}_{lay}", window=w, layout=lay)
    for h in [3, 10]:
        for target in ["relative", "absolute"]:
            add(f"15m_h{h}_{target}", frequency=15, horizon=h, target=target)
    add("wide_long_context", width=16, window=10, layout="rules_middle")
    add("wide_short_price", width=16, window=3, info="price_volume")
    return cases


def protocol(root):
    root = Path(root)
    spec = {"version": "timefolio-heatmap-pilot-v2-historical-designations", "source_start": "20250908", "source_end": END,
            "validation_start": VAL_START, "test_start": TEST_START,
            "purge": "label_end strictly earlier than next partition; whole dates stay together",
            "screen_seed": 17, "replicate_seeds": [17, 29, 43], "refine_top_n": 3,
            "configs": configurations(), "gbm_leaves": [7, 15, 31], "gbm_horizons": [3, 5, 10],
            "execution": "next session 09:05-09:34 typical-price VWAP proxy, 5% volume participation",
            "fees": {"buy": .001, "sell": .003, "slippage_per_side": .0005},
            "portfolio": {"initial_nav": 1e9, "long_only": True, "max_gross": .8, "stock_target": .08, "top_n": 12},
            "selection": "min(two validation-half annualised active means) - 0.25*annualised active volatility; minus 10 if four turnover violations",
            "reference": "5-session momentum, same execution and constraints",
            "holdout_policy": "Select one CNN ensemble and one GBM on validation only; refit through June with purged labels, then evaluate July-September once.",
            "rule_amendment": "Historical KRX designation intervals gate buys on execution date; individual weight drift triggers daily repair. Validation portfolios are recomputed before any holdout evaluation. Feature tensors and training labels remain unchanged.",
            "limitations": "Static current GICS, incomplete delistings, historical designation coverage is audited but not certified; no order-book replay; not contest certified."}
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Protocol already frozen; use a new research run rather than overwrite.")
    atomic_json(path, spec)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"protocol_sha256": digest, "cnn_configs": len(spec["configs"]), "gbm_configs": 9}), flush=True)
    return spec


def context(root):
    p = dict(np.load(root / "panel.npz"))
    p["eligible"] = p["eligible"].astype(bool); p["sector"] = p["sector"].astype(int)
    ix = json.loads((root / "panel_index.json").read_text())
    sam = np.load(root / "samples.npz")
    return p, ix, sam["ci"], sam["di"]


def purged_masks(dates, di, horizon, y):
    dates = np.asarray(dates)
    end = np.minimum(di + horizon, len(dates) - 1)
    valid = np.isfinite(y) & (di + horizon < len(dates))
    train = valid & (dates[end] < VAL_START)
    val = valid & (dates[di] >= VAL_START) & (dates[end] < TEST_START)
    refit = valid & (dates[end] < TEST_START)
    # Predictions include signals on the session before each evaluation period.
    val_pred = (di >= np.searchsorted(dates, VAL_START) - 1) & (dates[di] < TEST_START)
    test_pred = (di >= np.searchsorted(dates, TEST_START) - 1)
    return train, val, refit, val_pred, test_pred


def transform(images, cfg):
    a = np.array(images, copy=True)
    neutral = 128
    if cfg["info"] == "price_volume": a[:, 16:] = neutral
    if cfg["info"] == "no_rules": a[:, 24:] = neutral
    if cfg["info"] == "no_context": a[:, 16:24] = neutral
    if cfg["info"] == "summary_only":
        a[:, :8] = neutral
        a[:, 8:16] = np.repeat(a[:, 8:16, -1:], a.shape[-1], axis=2)
        a[:, 28:] = neutral
    layout = cfg["layout"]
    if layout == "rules_top": a = a[:, np.r_[24:32, 0:24]]
    if layout == "rules_middle": a = a[:, np.r_[0:8, 24:32, 8:24]]
    if layout == "interleave": a = a[:, np.arange(32).reshape(4, 8).T.ravel()]
    if layout == "rows_shuffled": a = a[:, np.random.default_rng(2718).permutation(32)]
    if layout == "time_shuffled": a = a[:, :, np.random.default_rng(3141).permutation(a.shape[2])]
    if layout == "transpose": a = a.transpose(0, 2, 1)
    return np.ascontiguousarray(a[:, None])


class HeatCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__(); w = cfg["width"]; k = tuple(cfg["kernel"])
        self.net = nn.Sequential(
            nn.Conv2d(1, w, k, padding=(k[0] // 2, k[1] // 2)), nn.LeakyReLU(.1), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.LeakyReLU(.1),
            nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Dropout(cfg["dropout"]), nn.Linear(32 * w, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def predict(model, x, indices, batch=512):
    model.eval(); result = []
    with torch.no_grad():
        for b in range(0, len(indices), batch):
            result.append(model(x[indices[b:b + batch]].float().div(127.5).sub(1)).sigmoid().cpu().numpy())
    return np.concatenate(result) if result else np.array([])


def fit_cnn(images, cfg, y, train, val, pred, seed, device, *, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)
    # uint8 resident tensors save host/device memory. Convert only each batch.
    x = torch.as_tensor(transform(images, cfg), device=device)
    yy = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=device)
    model = HeatCNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    tr, va, pr = np.where(train)[0], np.where(val)[0], np.where(pred)[0]
    best, state, best_epoch, history = -np.inf, None, 0, []
    for epoch in range(epochs or cfg["epochs"]):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(tr)
        losses = []
        for b in range(0, len(order), 256):
            ids = order[b:b + 256]
            opt.zero_grad(set_to_none=True)
            logits = model(x[ids].float().div(127.5).sub(1))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yy[ids])
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        ap = float(average_precision_score(y[va], predict(model, x, va))) if len(va) else 0.
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap": ap})
        if not len(va) or ap > best:
            best, best_epoch = ap, epoch + 1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    output = np.full(len(y), np.nan, dtype=np.float32); output[pr] = predict(model, x, pr)
    del x, model, opt
    if device == "cuda": torch.cuda.empty_cache()
    return output, {"best_epoch": best_epoch, "history": history, "seed": seed}, state


def score_matrix(prediction, ci, di, shape):
    out = np.full(shape, np.nan, dtype=np.float32); out[ci, di] = prediction
    return out


def validation_score(result, reference):
    a, b = pd.DataFrame(result["daily"]), pd.DataFrame(reference["daily"])
    r = a.nav.to_numpy() / np.r_[1e9, a.nav.to_numpy()[:-1]] - 1
    base = b.nav.to_numpy() / np.r_[1e9, b.nav.to_numpy()[:-1]] - 1
    active = r - base; halves = np.array_split(active, 2)
    value = min(x.mean() * 252 for x in halves) - .25 * active.std(ddof=1) * np.sqrt(252)
    if result["metrics"]["four_week_turnover_stop"]: value -= 10
    return float(value)


def prediction_metrics(y, pred, mask, di):
    valid = mask & np.isfinite(pred) & np.isfinite(y)
    auc = roc_auc_score(y[valid], pred[valid]) if len(np.unique(y[valid])) > 1 else .5
    return {"auc": float(auc), "ap": float(average_precision_score(y[valid], pred[valid])),
            "positive_rate": float(y[valid].mean()), "rows": int(valid.sum()), "dates": int(len(np.unique(di[valid])))}


def screen(root, device):
    spec = protocol(root); p, ix, ci, di = context(root)
    runs = root / "screen"; runs.mkdir(exist_ok=True)
    baseline = replay(p, ix, p["r5"], VAL_START, "20260630")
    atomic_json(runs / "momentum_reference.json", baseline)
    summary = []
    for cfg in spec["configs"]:
        out = runs / (cfg["id"] + ".json")
        if out.exists(): summary.append(json.loads(out.read_text())); continue
        started = time.monotonic()
        images = np.load(root / f"images_f{cfg['frequency']}_w{cfg['window']}.npy", mmap_mode="r")
        y, _ = labels(p, ci, di, cfg["horizon"], cfg["target"])
        train, val, _, val_pred, _ = purged_masks(ix["dates"], di, cfg["horizon"], y)
        pred, fit, state = fit_cnn(images, cfg, y, train, val, val_pred, 17, device)
        scores = score_matrix(pred, ci, di, p["close"].shape)
        bt = replay(p, ix, scores, VAL_START, "20260630", rebalance=cfg["rebalance"])
        result = {"config": cfg, "fit": fit, "prediction": prediction_metrics(y, pred, val, di),
                  "portfolio": bt["metrics"], "selection_score": validation_score(bt, baseline),
                  "seconds": round(time.monotonic() - started, 2)}
        atomic_json(out, result); np.save(runs / (cfg["id"] + ".pred.npy"), pred)
        torch.save(state, runs / (cfg["id"] + ".pt"))
        summary.append(result)
        print(json.dumps({"screen": cfg["id"], "score": result["selection_score"], "return": bt["metrics"]["return"], "seconds": result["seconds"]}), flush=True)
    atomic_json(root / "screen_summary.json", sorted(summary, key=lambda x: x["selection_score"], reverse=True))


def refine(root, device):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    spec = protocol(root); p, ix, ci, di = context(root)
    summary = json.loads((root / "screen_summary.json").read_text())
    reference = replay(p, ix, p["r5"], VAL_START, "20260630")
    dest = root / "refine"; dest.mkdir(exist_ok=True)
    refined = []
    for item in summary[:spec["refine_top_n"]]:
        cfg = item["config"]; out = dest / (cfg["id"] + ".json")
        if out.exists(): refined.append(json.loads(out.read_text())); continue
        images = np.load(root / f"images_f{cfg['frequency']}_w{cfg['window']}.npy", mmap_mode="r")
        y, _ = labels(p, ci, di, cfg["horizon"], cfg["target"])
        train, val, _, val_pred, _ = purged_masks(ix["dates"], di, cfg["horizon"], y)
        preds = [np.load(root / "screen" / (cfg["id"] + ".pred.npy"))]; fits = [item["fit"]]
        for seed in spec["replicate_seeds"][1:]:
            pred, fit, state = fit_cnn(images, cfg, y, train, val, val_pred, seed, device)
            preds.append(pred); fits.append(fit)
            torch.save(state, dest / f"{cfg['id']}_s{seed}.pt")
        avg = np.mean(preds, axis=0)
        bt = replay(p, ix, score_matrix(avg, ci, di, p["close"].shape), VAL_START, "20260630", rebalance=cfg["rebalance"])
        single = [replay(p, ix, score_matrix(v, ci, di, p["close"].shape), VAL_START, "20260630", rebalance=cfg["rebalance"])["metrics"] for v in preds]
        result = {"config": cfg, "fits": fits, "seed_metrics": single, "portfolio": bt["metrics"],
                  "selection_score": validation_score(bt, reference), "prediction": prediction_metrics(y, avg, val, di)}
        atomic_json(out, result); np.save(dest / (cfg["id"] + ".pred.npy"), avg); refined.append(result)
        print(json.dumps({"refined": cfg["id"], "score": result["selection_score"], "return": bt["metrics"]["return"]}), flush=True)
    # Same base heatmap information, flattened into last/mean/std feature summaries.
    images = np.load(root / "images_f30_w5.npy", mmap_mode="r")
    x = np.concatenate([images[:, :, -1], images.mean(2), images.std(2)], axis=1).astype(np.float32) / 255
    gbms = []
    for horizon in spec["gbm_horizons"]:
        y, _ = labels(p, ci, di, horizon)
        train, val, _, val_pred, _ = purged_masks(ix["dates"], di, horizon, y)
        for leaves in spec["gbm_leaves"]:
            cfg = {"horizon": horizon, "leaves": leaves}
            model = LGBMClassifier(n_estimators=300, num_leaves=leaves, learning_rate=.03,
                                   min_child_samples=100, reg_lambda=5, random_state=17, n_jobs=4, verbosity=-1)
            model.fit(x[train], y[train], eval_set=[(x[val], y[val])], callbacks=[early_stopping(25, verbose=False), log_evaluation(0)])
            pred = np.full(len(y), np.nan); pred[val_pred] = model.predict_proba(x[val_pred])[:, 1]
            bt = replay(p, ix, score_matrix(pred, ci, di, p["close"].shape), VAL_START, "20260630")
            result = {"config": cfg, "best_iteration": model.best_iteration_, "portfolio": bt["metrics"],
                      "selection_score": validation_score(bt, reference), "prediction": prediction_metrics(y, pred, val, di)}
            gbms.append(result)
            print(json.dumps({"gbm": cfg, "score": result["selection_score"]}), flush=True)
    selected = {"cnn": max(refined, key=lambda x: x["selection_score"]), "gbm": max(gbms, key=lambda x: x["selection_score"]),
                "cnn_refinement": refined, "gbm_screen": gbms, "holdout_opened": False}
    atomic_json(root / "selection.json", selected)


def holdout(root, device):
    from lightgbm import LGBMClassifier
    if (root / "holdout.json").exists():
        raise RuntimeError("Holdout already evaluated. Do not tune against it or overwrite it.")
    spec = protocol(root); p, ix, ci, di = context(root)
    selected = json.loads((root / "selection.json").read_text()); cfg = selected["cnn"]["config"]
    dest = root / "final_models"; dest.mkdir(exist_ok=True)
    images = np.load(root / f"images_f{cfg['frequency']}_w{cfg['window']}.npy", mmap_mode="r")
    y, _ = labels(p, ci, di, cfg["horizon"], cfg["target"])
    _, _, refit, _, test_pred = purged_masks(ix["dates"], di, cfg["horizon"], y)
    preds = []
    epochs = int(np.median([f["best_epoch"] for f in selected["cnn"]["fits"]]))
    for seed in spec["replicate_seeds"]:
        pred, fit, state = fit_cnn(images, cfg, y, refit, np.zeros(len(y), bool), test_pred, seed, device, epochs=epochs)
        preds.append(pred); torch.save({"config": cfg, "state_dict": state, "fit": fit}, dest / f"cnn_s{seed}.pt")
    cnn = score_matrix(np.mean(preds, axis=0), ci, di, p["close"].shape)
    gi = selected["gbm"]; gc = gi["config"]
    images = np.load(root / "images_f30_w5.npy", mmap_mode="r")
    x = np.concatenate([images[:, :, -1], images.mean(2), images.std(2)], axis=1).astype(np.float32) / 255
    gy, _ = labels(p, ci, di, gc["horizon"])
    _, _, gt, _, gp = purged_masks(ix["dates"], di, gc["horizon"], gy)
    model = LGBMClassifier(n_estimators=max(1, gi["best_iteration"]), num_leaves=gc["leaves"], learning_rate=.03,
                           min_child_samples=100, reg_lambda=5, random_state=17, n_jobs=4, verbosity=-1)
    model.fit(x[gt], gy[gt]); model.booster_.save_model(str(dest / "gbm.txt"))
    pred = np.full(len(gy), np.nan); pred[gp] = model.predict_proba(x[gp])[:, 1]
    gbm = score_matrix(pred, ci, di, p["close"].shape)
    scores = {"cnn": cnn, "gbm": gbm, "momentum5": p["r5"], "liquidity": p["adv20"]}
    results = {}
    for name, s in scores.items():
        period = cfg["rebalance"] if name == "cnn" else 5
        results[name] = replay(p, ix, s, TEST_START, END, rebalance=period, return_trades=True)
    stress = {}
    for name in ["cnn", "gbm"]:
        period = cfg["rebalance"] if name == "cnn" else 5
        stress[name] = {
            "slippage_10bp": replay(p, ix, scores[name], TEST_START, END, slip=.001, rebalance=period)["metrics"],
            "slippage_25bp": replay(p, ix, scores[name], TEST_START, END, slip=.0025, rebalance=period)["metrics"],
            "participation_1pct": replay(p, ix, scores[name], TEST_START, END, participation=.01, rebalance=period)["metrics"],
            "sector_floor_10pct": replay(p, ix, scores[name], TEST_START, END, sector_floor=True, rebalance=period)["metrics"]}
    atomic_json(root / "holdout.json", {"selected_cnn": cfg, "refit_epochs": epochs, "selected_gbm": gc,
                                        "paths": results, "stress": stress, "rule_certified": False})
    np.savez_compressed(root / "holdout_scores.npz", cnn=cnn, gbm=gbm)
    print(json.dumps({"holdout": {k: v["metrics"] for k, v in results.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["protocol", "screen", "refine", "holdout", "all"])
    ap.add_argument("--root", type=Path, default=ROOT); ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.device == "cuda": torch.backends.cudnn.benchmark = True
    if a.command == "protocol": protocol(a.root)
    if a.command in ("screen", "all"): screen(a.root, a.device)
    if a.command in ("refine", "all"): refine(a.root, a.device)
    if a.command in ("holdout", "all"): holdout(a.root, a.device)
