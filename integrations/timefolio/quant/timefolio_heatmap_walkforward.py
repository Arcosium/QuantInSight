"""Monthly expanding/rolling training for the next Timefolio heatmap research batch.

The previous July--September test has been observed. These are explicitly further
development results, never a fresh independent holdout. Every monthly model and
the online model selector nevertheless use only information available beforehand.
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from quant.timefolio_heatmap_data import ROOT, atomic_json
from quant.timefolio_heatmap_features import labels
from quant.timefolio_heatmap_study import context, HeatCNN, BASE, score_matrix

SOURCE = ROOT.with_name(ROOT.name + "_rules_v2")
DEST = ROOT.with_name(ROOT.name + "_walkforward_v3")


def configurations():
    base = dict(architecture="cnn", encoding="raw", target="sector", horizon=3,
                train_window=0, epochs=6, lr=.0007, seed=17)
    cases = []
    def add(name, **kw): cases.append(dict(base, id=name, **kw))
    add("monthly_binary", target="binary", horizon=5)
    add("cnn_sector_h3")
    add("cnn_sector_h5", horizon=5)
    add("cnn_center_h3", encoding="sector_center")
    add("cnn_center_h5", encoding="sector_center", horizon=5)
    add("cnn_vol_h3", encoding="vol_scaled")
    add("cnn_absolute_h3", target="absolute")
    add("split_sector_h3", architecture="split")
    add("split_center_h3", architecture="split", encoding="sector_center")
    add("split_rank_h5", architecture="split", encoding="sector_center", target="sector_rank", horizon=5)
    add("split_absolute_h3", architecture="split", target="absolute")
    add("tcn_sector_h3", architecture="tcn")
    add("tcn_center_h3", architecture="tcn", encoding="sector_center")
    add("tcn_vol_recent", architecture="tcn", encoding="vol_scaled", train_window=60, horizon=5)
    add("tcn_absolute_recent", architecture="tcn", target="absolute", train_window=60)
    add("cnn_center_recent", encoding="sector_center", train_window=60)
    add("mlp_sector_h3", architecture="mlp")
    add("mlp_absolute_h3", architecture="mlp", target="absolute")
    add("gbm_sector_h3", architecture="gbm")
    add("gbm_sector_h5", architecture="gbm", horizon=5)
    add("gbm_absolute_h3", architecture="gbm", target="absolute")
    add("gbm_rank_h5", architecture="gbm", target="sector_rank", horizon=5)
    return cases


def freeze_protocol(dest=DEST, source=SOURCE):
    dest, source = Path(dest), Path(source); dest.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": "timefolio-walkforward-v3", "source": str(source),
        "source_period": ["20250908", "20260923"], "evaluation_start": "20260101",
        "evaluation_end": "20260923", "origins": "first exchange session each calendar month",
        "inner_validation_sessions": 20, "label_purge": "label end < next partition origin",
        "refit": "select epoch using past inner validation, then restart and fit all observed labels",
        "configs": configurations(), "common_selector_target": "3-session sector-residual return",
        "online_selection": "choose top one or three architectures by common past inner daily Spearman IC",
        "execution": "next session 09:05--09:34; 5% participation; at most 20 filled orders/day, deferred requests retried",
        "policies": [
            {"id": "sector20", "sector_cap": .20, "stock_weight": .05, "top_n": 20, "regime": "none"},
            {"id": "sector15", "sector_cap": .15, "stock_weight": .04, "top_n": 24, "regime": "none"},
            {"id": "sector20_trend", "sector_cap": .20, "stock_weight": .05, "top_n": 20, "regime": "trend"},
            {"id": "sector20_vol", "sector_cap": .20, "stock_weight": .05, "top_n": 20, "regime": "volatility"}],
        "regime_rules": "trend: gross 0.8 when broad market proxy >= trailing SMA20, else 0.4; volatility: clip(0.12/annualised trailing 20-day vol, 0.3, 0.8)",
        "benchmarks": ["momentum5", "momentum20", "reversal1", "reversal5", "sector_reversal5", "lowvol", "liquidity", "online_nonimage"],
        "statistics": "paired date-block bootstrap; max statistic across neural configurations and policies; blocks 5 and 10; cash and same-policy online nonimage comparator",
        "candidate_gate": "net return > 0, positive in >=2 of 3 calendar blocks, MDD > -20%, turnover stop false, family-adjusted evidence versus nonimage comparator; seed/stress replication then independent confirmation still required",
        "status": "research development; old holdout already observed; no live orders or automatic promotion",
        "limitations": "current GICS; survivor universe; inferred corporate actions; order book/queue delay approximated, daily order budget is conservative assumption, not exact legal certification"}
    path = dest / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError("Frozen specification differs; create a new study version")
    if not path.exists(): atomic_json(path, spec)
    return spec


def continuous_targets(p, ci, di, horizon, mode):
    binary, forward = labels(p, ci, di, horizon)
    if mode == "binary": return binary
    frame = pd.DataFrame({"date": di, "sector": p["sector"][ci], "return": forward})
    if mode == "absolute":
        out = forward - .004
    else:
        groups = frame.groupby(["date", "sector"])["return"]
        size = groups.transform("count").to_numpy()
        if mode == "sector_rank":
            local = groups.rank(pct=True).to_numpy()
            market = frame.groupby("date")["return"].rank(pct=True).to_numpy()
            return np.where(size >= 4, local, market).astype(np.float32) * 2 - 1
        center = groups.transform("median").to_numpy()
        fallback = frame.groupby("date")["return"].transform("median").to_numpy()
        out = forward - np.where(size >= 4, center, fallback)
    return np.clip(out / .10, -3, 3).astype(np.float32)


def encode_images(raw, ci, di, sectors, vol, encoding):
    a = np.array(raw, dtype=np.uint8, copy=True)
    if encoding == "raw": return a
    if encoding == "vol_scaled":
        scale = np.clip(.02 / np.maximum(vol[ci, di], .005), .25, 4)
        x = (a[:, :8].astype(np.float32) / 127.5 - 1) * scale[:, None, None]
        a[:, :8] = np.rint((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8)
        return a
    if encoding != "sector_center": raise ValueError(encoding)
    chosen_rows = np.r_[0:14, 16:22]
    # Group only by the same signal date and sector. No statistics pool future days.
    for d in np.unique(di):
        same_day = np.flatnonzero(di == d)
        day_median = np.median(raw[same_day][:, chosen_rows], axis=0)
        for sec in np.unique(sectors[ci[same_day]]):
            ids = same_day[sectors[ci[same_day]] == sec]
            center = np.median(raw[ids][:, chosen_rows], axis=0) if len(ids) >= 4 else day_median
            x = raw[ids][:, chosen_rows].astype(np.float32) - center + 128
            x = np.clip(x, 0, 255).astype(np.uint8)
            # Preserve absent intraday prints as neutral pixels.
            missing = raw[ids, 14] <= 128
            x[:, :14] = np.where(missing[:, None, :], 128, x[:, :14])
            for j, row in enumerate(chosen_rows): a[ids, row] = x[:, j]
    return a


def prepare(dest=DEST, source=SOURCE):
    dest, source = Path(dest), Path(source); spec = freeze_protocol(dest, source)
    p, ix, ci, di = context(source)
    raw = np.load(source / "images_f30_w5.npy", mmap_mode="r")
    for encoding in sorted({c["encoding"] for c in spec["configs"]}):
        target = dest / f"images_{encoding}.npy"
        if target.exists(): continue
        if encoding == "raw":
            import os
            os.link(source / "images_f30_w5.npy", target)
        else:
            np.save(target, encode_images(raw, ci, di, p["sector"], p["vol20"], encoding))
        print(json.dumps({"prepared_encoding": encoding}), flush=True)
    artifacts = [source / "panel.npz", source / "samples.npz", source / "panel_index.json", dest / "protocol.json"]
    artifacts += list(dest.glob("images_*.npy"))
    hashes = {}
    for path in artifacts:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1048576), b""): h.update(b)
        hashes[str(path)] = h.hexdigest()
    manifest = dest / "data_hashes.json"
    if manifest.exists() and json.loads(manifest.read_text()) != hashes: raise RuntimeError("Data changed")
    if not manifest.exists(): atomic_json(manifest, hashes)


def monthly_folds(dates, first="20260101"):
    a = np.asarray(dates); months = sorted({d[:6] for d in dates if d >= first})
    folds = []
    for month in months:
        ids = np.flatnonzero(np.char.startswith(a, month))
        folds.append({"month": month, "start": int(ids[0]), "end": int(ids[-1] + 1)})
    return folds


def fold_masks(di, y, allowed, horizon, start, end, train_window=0, validation_sessions=20):
    observed = np.isfinite(y) & allowed & (di + horizon < start)
    inner = start - validation_sessions
    lower = max(0, start - train_window) if train_window else 0
    train = np.isfinite(y) & allowed & (di + horizon < inner) & (di >= lower)
    val = observed & (di >= inner)
    refit = observed & (di >= lower)
    # The final session of a month is scored by the next month's refitted model.
    pred = (di >= start - 1) & (di < end - 1)
    inner_pred = (di >= inner) & (di < start)
    return train, val, refit, pred, inner_pred


def daily_ic(y, pred, mask, di):
    ok = mask & np.isfinite(y) & np.isfinite(pred)
    values = []
    for d in np.unique(di[ok]):
        use = ok & (di == d)
        if use.sum() >= 10 and np.std(y[use]) > 1e-8 and np.std(pred[use]) > 1e-8:
            values.append(float(spearmanr(y[use], pred[use]).statistic))
    return float(np.mean(values)) if values else 0.


class AlternativeNet(nn.Module):
    def __init__(self, architecture):
        super().__init__(); self.architecture = architecture
        if architecture == "cnn": self.full = HeatCNN(dict(BASE))
        elif architecture == "split":
            self.price = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                                       nn.Conv2d(8, 16, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                                       nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten())
            self.summary = nn.Sequential(nn.Linear(32, 32), nn.LeakyReLU(.1))
            self.head = nn.Sequential(nn.Linear(160, 32), nn.LeakyReLU(.1), nn.Dropout(.1), nn.Linear(32, 1))
        elif architecture == "tcn":
            self.temporal = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.LeakyReLU(.1),
                                          nn.Conv1d(16, 16, 3, padding=2, dilation=2), nn.LeakyReLU(.1),
                                          nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(.1), nn.Linear(64, 1))
        elif architecture == "mlp":
            self.summary = nn.Sequential(nn.Linear(96, 32), nn.LeakyReLU(.1), nn.Dropout(.1),
                                         nn.Linear(32, 16), nn.LeakyReLU(.1), nn.Linear(16, 1))
        else: raise ValueError(architecture)

    def forward(self, x):
        if self.architecture == "cnn": return self.full(x)
        if self.architecture == "tcn": return self.temporal(x[:, 0]).squeeze(1)
        if self.architecture == "split":
            a = x[:, 0, 16:]; summary = torch.cat([a.mean(-1), a.std(-1, correction=0)], 1)
            return self.head(torch.cat([self.price(x[:, :, :16]), self.summary(summary)], 1)).squeeze(1)
        a = x[:, 0]; summary = torch.cat([a[:, :, -1], a.mean(-1), a.std(-1, correction=0)], 1)
        return self.summary(summary).squeeze(1)


def predict(model, x, indices, binary=False):
    model.eval(); pieces = []
    with torch.no_grad():
        for k in range(0, len(indices), 512):
            out = model(x[indices[k:k + 512]].float().div(127.5).sub(1))
            if binary: out = out.sigmoid()
            pieces.append(out.numpy())
    return np.concatenate(pieces) if pieces else np.array([], np.float32)


def train_nn(x, y, train, val, cfg, di, *, epochs=None):
    seed = cfg["seed"]; torch.manual_seed(seed)
    model = AlternativeNet(cfg["architecture"]); opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=.001)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    binary = cfg["target"] == "binary"; best = -np.inf; state = None; best_epoch = 0; history = []
    for epoch in range(epochs or cfg["epochs"]):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(tr); losses = []
        for j in range(0, len(order), 256):
            ids = order[j:j + 256]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1))
            loss = (nn.functional.binary_cross_entropy_with_logits(out, target[ids]) if binary
                    else nn.functional.huber_loss(out, target[ids], delta=1.))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); losses.append(float(loss.detach()))
        metric = 0.
        if len(va):
            vp = np.full(len(y), np.nan); vp[va] = predict(model, x, va, binary)
            metric = (float(average_precision_score(y[va], vp[va])) if binary else daily_ic(y, vp, val, di))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "inner_metric": metric})
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"best_epoch": best_epoch, "history": history, "training_rows": len(tr), "validation_rows": len(va)}


def run(dest=DEST, source=SOURCE, *, only=None):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    dest, source = Path(dest), Path(source); spec = freeze_protocol(dest, source)
    p, ix, ci, di = context(source); dates = np.asarray(ix["dates"])
    allowed = np.zeros(len(ci), bool); valid_next = di + 1 < len(dates)
    allowed[valid_next] = p["trade_allowed"][ci[valid_next], di[valid_next] + 1].astype(bool)
    common_y = continuous_targets(p, ci, di, 3, "sector")
    for cfg in spec["configs"]:
        if only and cfg["id"] not in only: continue
        folder = dest / "models" / cfg["id"]; folder.mkdir(parents=True, exist_ok=True)
        raw = np.load(dest / f"images_{cfg['encoding']}.npy", mmap_mode="r")
        if cfg["architecture"] == "gbm":
            x = np.concatenate([raw[:, :, -1], raw.mean(2), raw.std(2)], 1).astype(np.float32) / 255
        else: x = torch.from_numpy(np.array(raw[:, None], copy=True))
        y = continuous_targets(p, ci, di, cfg["horizon"], cfg["target"])
        for fold in monthly_folds(ix["dates"], spec["evaluation_start"]):
            path = folder / f"{fold['month']}.json"
            if path.exists():
                if not path.with_suffix(".pred.npy").exists(): raise RuntimeError("Incomplete checkpoint")
                continue
            started = time.monotonic(); start, end = fold["start"], fold["end"]
            train, val, refit, pred, inner_pred = fold_masks(di, y, allowed, cfg["horizon"], start, end, cfg["train_window"])
            if train.sum() < 1000 or val.sum() < 200: raise RuntimeError("Insufficient causal training data")
            common_mask = inner_pred & allowed & (di + 3 < start) & np.isfinite(common_y)
            inner_scores = np.full(len(ci), np.nan, np.float32)
            if cfg["architecture"] == "gbm":
                def estimator(n):
                    return LGBMRegressor(n_estimators=n, num_leaves=15, learning_rate=.03, min_child_samples=100,
                                         reg_lambda=10., subsample=.8, subsample_freq=1, colsample_bytree=.8,
                                         random_state=cfg["seed"], n_jobs=4, verbosity=-1)
                model = estimator(250)
                model.fit(x[train], y[train], eval_X=x[val], eval_y=y[val], callbacks=[early_stopping(25, verbose=False), log_evaluation(0)])
                inner_scores[inner_pred] = model.predict(x[inner_pred])
                fit = {"best_epoch": max(1, int(model.best_iteration_)), "training_rows": int(train.sum()), "validation_rows": int(val.sum())}
                model = estimator(fit["best_epoch"]); model.fit(x[refit], y[refit])
                values = model.predict(x[pred]); model.booster_.save_model(str(folder / f"{fold['month']}.txt"))
            else:
                model, fit = train_nn(x, y, train, val, cfg, di)
                inner_scores[inner_pred] = predict(model, x, np.flatnonzero(inner_pred), cfg["target"] == "binary")
                del model
                model, _ = train_nn(x, y, refit, np.zeros(len(y), bool), cfg, di, epochs=fit["best_epoch"])
                values = predict(model, x, np.flatnonzero(pred), cfg["target"] == "binary")
                torch.save({"config": cfg, "state_dict": model.state_dict()}, folder / f"{fold['month']}.pt")
            out = np.full(len(ci), np.nan, np.float32); out[pred] = values
            np.save(path.with_suffix(".pred.npy"), out)
            row = {"config": cfg, "fold": fold, "fit": fit,
                   "common_inner_ic": daily_ic(common_y, inner_scores, common_mask, di),
                   "refit_rows": int(refit.sum()), "last_refit_label": str(max(dates[di[refit] + cfg["horizon"]])),
                   "first_execution": ix["dates"][start], "last_execution": ix["dates"][end - 1],
                   "seconds": round(time.monotonic() - started, 2)}
            atomic_json(path, row)
            print(json.dumps({"model": cfg["id"], "month": fold["month"], "epochs": fit["best_epoch"],
                              "inner_ic": row["common_inner_ic"], "seconds": row["seconds"]}), flush=True)
            del model
        del x
    print(json.dumps({"training_complete": True}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["prepare", "run", "all"])
    ap.add_argument("--source", type=Path, default=SOURCE); ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.command in ("prepare", "all"): prepare(a.dest, a.source)
    if a.command in ("run", "all"): run(a.dest, a.source, only=a.only)
