"""KRX price-image CNN: fixed experiment, next-session execution and baselines.

Inspired by Jiang/Kelly/Xiu's price-image representation, not a replication.
No broker is imported. Artifacts and paper account live in the private vault.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from arcmarket.systematic import features, replay, select
from quant.krx_cnn_data import atomic_json, load, refresh

ARTIFACTS = Path(os.getenv("ARCTRADE_CNN_DIR", Path.home()/"vault"/"ArcTrade"/"krx_cnn"))
SPEC = {"version": "krx-image-v1", "window": 60, "horizon": 5, "seed": 17,
        "validation_start": "2025-06-01", "test_start": "2025-09-01", "epochs": 4,
        "train_stride": 3, "rebalance_sessions": 5, "per_side_cost": .002,
        "max_stock_weight": .08, "max_gross_weight": .8,
        "label": "next open to fifth close, relative to eligible-universe median",
        "selection": "CNN probability > 0.5; trend/liquidity filters; top 10",
        "promotion": "paper research only; no automatic broker promotion"}


def image_windows(windows):
    """60 OHLC bars and volume, scaled using only that window, 48 x 180."""
    a = np.asarray(windows, dtype=np.float32)
    n, width, _ = a.shape
    out = np.zeros((n, 1, 48, width*3), dtype=np.uint8)
    lo, hi = a[:, :, 2].min(1), a[:, :, 1].max(1)
    scale = 35/np.maximum(hi-lo, 1e-6)
    y = np.clip(np.rint((hi[:, None, None]-a[:, :, :4])*scale[:, None, None]), 0, 35).astype(int)
    vmax = np.maximum(a[:, :, 4].max(1), 1)
    vh = np.rint(a[:, :, 4]/vmax[:, None]*10).astype(int)
    idx = np.arange(n)
    for j in range(width):
        for row in range(36):
            out[:, 0, row, j*3+1] = ((row >= y[:, j, 1]) & (row <= y[:, j, 2]))*255
        out[idx, 0, y[:, j, 0], j*3] = 255
        out[idx, 0, y[:, j, 3], j*3+2] = 255
        for row in range(10):
            out[:, 0, 47-row, j*3+1] = (row < vh[:, j])*255
    return out


class PriceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(), nn.Linear(16*3*5, 32), nn.ReLU(),
            nn.Dropout(.15), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def samples(histories, frame, *, training=True):
    calendar = pd.DatetimeIndex(sorted(frame.date.unique()))
    allowed = frame[frame.valid & (frame.adv >= 1e9)]
    keys = set(zip(allowed.code, allowed.date))
    images, rows = [], []
    for code, raw in histories.items():
        d = raw.reindex(calendar)
        a = d.to_numpy(np.float32)
        batch, meta = [], []
        for i in range(60, len(d)):
            date = d.index[i]
            if (code, date) not in keys:
                continue
            if training and date < pd.Timestamp(SPEC["validation_start"]) and i % SPEC["train_stride"]:
                continue
            win = a[i-59:i+1]
            if not np.isfinite(win).all() or (win[:, :4] <= 0).any():
                continue
            row = {"code": code, "date": date, "label_end": pd.NaT, "fwd": np.nan}
            if i+5 < len(d):
                future = a[i+1:i+6]
                if np.isfinite(future).all() and (future[:, :4] > 0).all() and (future[:, 4] > 0).all():
                    row.update(label_end=d.index[i+5], fwd=float(future[-1, 3]/future[0, 0]-1))
            batch.append(win); meta.append(row)
        if batch:
            images.append(image_windows(batch)); rows.extend(meta)
    if not images:
        raise ValueError("no_valid_price_windows")
    meta = pd.DataFrame(rows)
    med = meta.groupby("date").fwd.transform("median")
    meta["label"] = (meta.fwd > med).astype(np.float32)
    return np.concatenate(images), meta


def purged_masks(meta):
    v, t = pd.Timestamp(SPEC["validation_start"]), pd.Timestamp(SPEC["test_start"])
    valid = meta.fwd.notna() & meta.label_end.notna()
    return (valid & (meta.date < v) & (meta.label_end < v),
            valid & (meta.date >= v) & (meta.date < t) & (meta.label_end < t),
            meta.date >= t)


def predict(model, images):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), 256):
            x = torch.from_numpy(images[i:i+256]).float()/255
            out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out) if out else np.array([])


def fit(images, meta):
    torch.manual_seed(SPEC["seed"])
    np.random.seed(SPEC["seed"])
    torch.set_num_threads(4)
    tr, va, _ = purged_masks(meta)
    if tr.sum() < 1000 or va.sum() < 500:
        raise ValueError("insufficient_purged_training_history")
    net = PriceCNN()
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    train = TensorDataset(torch.from_numpy(images[tr]), torch.from_numpy(meta.loc[tr, "label"].to_numpy()))
    loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
    best, loss_best, history = None, float("inf"), []
    for epoch in range(SPEC["epochs"]):
        start = time.time(); net.train(); losses = []
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x.float()/255), y)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        p = predict(net, images[va]); y = meta.loc[va, "label"].to_numpy()
        vl = float(-(y*np.log(p.clip(1e-7, 1))+(1-y)*np.log((1-p).clip(1e-7, 1))).mean())
        row = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation_loss": vl,
               "seconds": round(time.time()-start, 1)}
        history.append(row); print(json.dumps(row), flush=True)
        if vl < loss_best:
            best, loss_best = copy.deepcopy(net.state_dict()), vl
    net.load_state_dict(best)
    return net, history, {"train": int(tr.sum()), "validation": int(va.sum()),
                          "last_train_label": str(meta.loc[tr, "label_end"].max().date()),
                          "last_validation_label": str(meta.loc[va, "label_end"].max().date())}


def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Spec is written before evaluation. No selection on the test-period P&L.
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    run_dir = ARTIFACTS/"runs"/stamp
    run_dir.mkdir(parents=True)
    atomic_json(run_dir/"spec.json", SPEC)
    histories = load(); frame = features(histories)
    images, meta = samples(histories, frame)
    print(json.dumps({"phase": "prepared", "histories": len(histories), "samples": len(meta)}), flush=True)
    model, epochs, split = fit(images, meta)
    _, _, test = purged_masks(meta)
    tm = meta.loc[test, ["code", "date"]].copy()
    tm["p"] = predict(model, images[test])
    scores = {t: dict(zip(g.code, g.p)) for t, g in tm.groupby("date")}
    end = frame.date.max()
    paths = {}
    for method in ("cnn", "momentum", "equal_weight"):
        paths[method] = replay(histories, frame, start=SPEC["test_start"], end=end,
                               scores=scores if method=="cnn" else None, method=method)
    stress = replay(histories, frame, start=SPEC["test_start"], end=end, scores=scores,
                    fee=.004)["metrics"]
    baseline = max((paths[k]["metrics"]["return_pct"] or 0 for k in ("momentum", "equal_weight")))
    m = paths["cnn"]["metrics"]
    historical_pass = bool(all(p["metrics"]["valid"] for p in paths.values()) and m["return_pct"] > max(0, baseline) and
                           (stress["return_pct"] or -100) > 0 and m["mdd_pct"] > -20)
    model_path = run_dir/"model.pt"
    torch.save(model.state_dict(), model_path)
    model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    tm.to_parquet(run_dir/"test_predictions.parquet", index=False)
    report = {"spec": SPEC, "model_id": model_id, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
              "split": split, "epochs": epochs, "start": SPEC["test_start"], "end": str(end.date()),
              "paths": paths, "stress_double_cost": stress, "historical_pass": historical_pass,
              "mode": "forward_paper", "promotion": "unvalidated_no_broker",
              "limitations": ["현재 유니버스를 과거에 적용한 생존편향", "배당·상장폐지·기업행동 미검증",
                  "공급자 차트 기준 가격; 과거 시점 데이터 원본 아님", "시가 체결·비용은 가정, 미체결 위험 존재",
                  "이번 비교 결과를 보고 규칙을 바꾸면 해당 구간은 검증용으로 재사용 불가"]}
    atomic_json(run_dir/"report.json", report)
    atomic_json(ARTIFACTS/"current.json", {"run_dir": str(run_dir), "model_id": model_id})
    print(json.dumps({k: v["metrics"] for k, v in paths.items()}), flush=True)
    return report


def latest_signal():
    torch.set_num_threads(2)
    current = json.loads((ARTIFACTS/"current.json").read_text())
    model = PriceCNN()
    model.load_state_dict(torch.load(Path(current["run_dir"])/"model.pt", weights_only=True, map_location="cpu"))
    histories = load(); frame = features(histories)
    day = frame.date.max()
    latest = frame[frame.date == day]
    coverage = len(latest)/max(1, len(histories))
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if coverage < .95 or (now.tz_localize(None).normalize()-day).days > 4:
        return {"status": "stale_or_incomplete", "as_of": str(day.date()), "coverage": coverage,
                "weights": {}, "rows": [], "model_id": current["model_id"]}
    windows, codes = [], []
    for code, d in histories.items():
        d = d[d.index <= day].tail(60)
        if len(d)==60 and d.index[-1]==day and np.isfinite(d.to_numpy()).all() and (d.volume > 0).all():
            windows.append(d.to_numpy()); codes.append(code)
    p = predict(model, image_windows(windows)) if windows else []
    plan = select(latest, scores=dict(zip(codes, p)))
    return {**plan, "status": "ready", "as_of": str(day.date()), "coverage": coverage,
            "created_at": now.isoformat(), "model_id": current["model_id"], "mode": "forward_paper"}


def audit():
    """Revalue persisted predictions after accounting/data corrections only.

    Keep the original report; no parameter fitting or checkpoint selection.
    """
    current = json.loads((ARTIFACTS/"current.json").read_text())
    root = Path(current["run_dir"])
    report = json.loads((root/"report.json").read_text())
    histories = load(); frame = features(histories)
    tm = pd.read_parquet(root/"test_predictions.parquet")
    scores = {t: dict(zip(g.code, g.p)) for t, g in tm.groupby("date")}
    paths = {method: replay(histories, frame, start=report["start"], end=report["end"],
                           scores=scores if method=="cnn" else None, method=method)
             for method in ("cnn", "momentum", "equal_weight")}
    stress = replay(histories, frame, start=report["start"], end=report["end"], scores=scores, fee=.004)["metrics"]
    m = paths["cnn"]["metrics"]
    passed = (all(p["metrics"]["valid"] for p in paths.values()) and m["return_pct"] > 0
              and all(m["return_pct"] > paths[k]["metrics"]["return_pct"] for k in ("momentum", "equal_weight"))
              and (stress["return_pct"] or -100)>0 and m["mdd_pct"] > -20)
    report.update(paths=paths, stress_double_cost=stress, historical_pass=bool(passed),
                  accounting_revision="halt close marked without trading; missing benchmarks block promotion")
    name="audit_"+pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")+".json"
    atomic_json(root/name, report)
    atomic_json(ARTIFACTS/"current.json", dict(current, report_file=name))
    print(json.dumps({k:p["metrics"] for k,p in paths.items()}), flush=True)
    # QuantInSight uses the separate deterministic policy, at 40% gross cap.
    baseline={k:replay(histories,frame,start=report["start"],end=report["end"],method=k,weight_scale=.5)
              for k in ("momentum","equal_weight")}
    qreport={"version":"trend-rank-v1", "scope":"QuantInSight quantitative research desk only; committee and broker not replayed",
             "mode":"paper_pilot", "start":report["start"], "end":report["end"], "paths":baseline,
             "max_stock_budget":.4, "limitations":report["limitations"], "validated":False}
    qpath=Path.home()/"vault"/"QuantInSight"/"research"/"stock_policy_baseline.json"
    atomic_json(qpath,qreport)
    print(json.dumps({"qis_research_only":{k:v["metrics"] for k,v in baseline.items()}}),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["train", "signal", "refresh", "audit"])
    args = ap.parse_args()
    if args.command == "train":
        train()
    elif args.command == "audit":
        audit()
    elif args.command == "refresh":
        print(json.dumps(refresh(), ensure_ascii=False))
    else:
        atomic_json(ARTIFACTS/"signal.json", latest_signal())
