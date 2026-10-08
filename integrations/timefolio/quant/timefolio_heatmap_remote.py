"""Own one inexpensive research Pod, recover artifacts, then terminate it.

The Runpod API credential stays on the local host. Only an explicit allowlist of
market tensors and source modules is uploaded. Other Pods are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from quant.timefolio_heatmap_data import ROOT, atomic_json

PROJECT = Path(__file__).resolve().parents[1]
NAME = "arctrade-timefolio-heatmap-20260928"
IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404"


def api():
    sys.path.insert(0, str(Path.home() / "projects/HYFE_QTPA"))
    from hyfe.runpod import rest
    return rest


def pack(root):
    files = ["panel.npz", "panel_index.json", "samples.npz", "protocol.json", "data_audit.json"]
    files += [f"images_f{f}_w{w}.npy" for f in (15, 30) for w in (3, 5, 10)]
    archive = root / "training_export.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for name in files:
            tar.add(root / name, arcname="data/" + name)
        for name in ["timefolio_heatmap_data.py", "timefolio_heatmap_features.py",
                     "timefolio_heatmap_replay.py", "timefolio_heatmap_study.py"]:
            tar.add(PROJECT / "quant" / name, arcname="quant/" + name)
    return archive


def run(root, max_hours=2.):
    rest = api(); root.mkdir(parents=True, exist_ok=True)
    receipt = root / "runpod_receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("state") != "terminated":
            raise RuntimeError("A research Pod receipt already exists; reconcile it before renting another.")
    # Prepare transfer before billing starts.
    archive = pack(root)
    key = root / "research_ssh_key"
    if not key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    public_key = key.with_suffix(".pub").read_text().strip()
    candidates = [("NVIDIA RTX A5000", "COMMUNITY", .19),
                  ("NVIDIA RTX 4000 SFF Ada Generation", "COMMUNITY", .22),
                  ("NVIDIA RTX A4500", "COMMUNITY", .23),
                  ("NVIDIA RTX 4000 Ada Generation", "COMMUNITY", .24),
                  ("NVIDIA GeForce RTX 3090", "COMMUNITY", .26),
                  ("NVIDIA RTX 2000 Ada Generation", "SECURE", .28),
                  ("NVIDIA RTX A4500", "SECURE", .29),
                  ("NVIDIA RTX A4000", "SECURE", .29),
                  ("NVIDIA RTX A5000", "SECURE", .31),
                  ("NVIDIA RTX 4000 Ada Generation", "SECURE", .32),
                  ("NVIDIA GeForce RTX 4090", "COMMUNITY", .39),
                  ("NVIDIA RTX A4000", "COMMUNITY", .24),
                  ("NVIDIA L4", "SECURE", .53),
                  ("NVIDIA A40", "SECURE", .53)]
    record, pod = None, None
    try:
        for gpu, cloud, ceiling in candidates:
            try:
                pod = rest("POST", "/pods", {
                    "name": NAME, "imageName": IMAGE, "cloudType": cloud,
                    "gpuCount": 1, "gpuTypeIds": [gpu], "containerDiskInGb": 30,
                    "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
                    "env": {"PUBLIC_KEY": public_key}, "minRAMPerGPU": 16, "minVCPUPerGPU": 2})
            except Exception as exc:
                print(json.dumps({"unavailable": gpu, "cloud": cloud, "reason": str(exc)[:350]}), flush=True)
                continue
            record = {"id": pod["id"], "name": NAME, "gpu": gpu, "cloud": cloud,
                      "cost_per_hour": pod.get("costPerHr"), "started_at": time.time(), "state": "starting"}
            atomic_json(receipt, record)
            # Storage is included in costPerHr; reject an unexpectedly expensive host.
            if not pod.get("costPerHr") or pod["costPerHr"] > ceiling + .015:
                raise RuntimeError("Pod quote exceeded the declared ceiling")
            print(json.dumps({k: record[k] for k in ["gpu", "cloud", "cost_per_hour", "state"]}), flush=True)
            break
        if pod is None:
            raise RuntimeError("No affordable research GPU available")
        deadline = record["started_at"] + max_hours * 3600
        connection = None
        ssh_base = ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={root / 'known_hosts'}"]
        for _ in range(36):
            status = rest("GET", "/pods/" + pod["id"])
            port = (status.get("portMappings") or {}).get("22")
            ip = status.get("publicIp")
            if ip and port:
                connection = (ip, str(port))
                probe = subprocess.run(ssh_base + ["-p", str(port), "root@" + ip, "true"], capture_output=True, timeout=20)
                if probe.returncode == 0: break
            time.sleep(5)
        else:
            raise TimeoutError("Research Pod did not become SSH-ready within three minutes")
        ip, port = connection
        ssh = ssh_base + ["-p", port, "root@" + ip]
        scp = ["scp", "-i", str(key), "-P", port, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-o", f"UserKnownHostsFile={root / 'known_hosts'}"]
        subprocess.run(scp + [str(archive), f"root@{ip}:/tmp/training.tar.gz"], check=True, timeout=300)
        setup = ("mkdir -p /workspace/study && cd /workspace/study && tar -xzf /tmp/training.tar.gz && "
                 "python -m pip install -q numpy pandas pyarrow scikit-learn lightgbm && "
                 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
                 "(nohup bash -c 'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m quant.timefolio_heatmap_study all "
                 "--root /workspace/study/data --device cuda > train.log 2>&1; echo $? > exit_code' "
                 "< /dev/null > launcher.log 2>&1 &)")
        subprocess.run(ssh + [setup], check=True, timeout=300)
        record["state"] = "training"; atomic_json(receipt, record)
        while time.time() < deadline:
            check = subprocess.run(ssh + ["cd /workspace/study && if test -f exit_code; then cat exit_code; else echo RUNNING; fi"],
                                   capture_output=True, text=True, timeout=30)
            if check.returncode == 0 and check.stdout.strip() != "RUNNING":
                record["training_exit"] = check.stdout.strip(); break
            status = subprocess.run(ssh + ["cd /workspace/study && tail -2 train.log"], capture_output=True, text=True, timeout=30)
            print(status.stdout.strip(), flush=True)
            time.sleep(30)
        else:
            record["training_exit"] = "deadline"
        # Recover even partial results or exception logs before destroying this Pod.
        export = ("cd /workspace/study && tar -czf /tmp/results.tar.gz "
                  "--exclude='data/images*' --exclude='data/panel.npz' --exclude='data/samples.npz' "
                  "data train.log launcher.log")
        subprocess.run(ssh + [export], check=True, timeout=120)
        subprocess.run(scp + [f"root@{ip}:/tmp/results.tar.gz", str(root / "runpod_results.tar.gz")], check=True, timeout=180)
        record["recovered"] = True
        atomic_json(receipt, record)
    finally:
        if pod is not None:
            owned = rest("GET", "/pods/" + pod["id"])
            if owned.get("name") != NAME:
                raise RuntimeError("Pod ownership mismatch; refusing termination")
            rest("DELETE", "/pods/" + pod["id"])
            record["finished_at"] = time.time(); record["state"] = "terminated"
            record["estimated_cost_usd"] = (record["finished_at"] - record["started_at"]) / 3600 * record["cost_per_hour"]
            # Verify deletion instead of inferring it from a successful request.
            remaining = rest("GET", "/pods")
            record["deletion_verified"] = not any(x["id"] == pod["id"] for x in remaining)
            atomic_json(receipt, record)
            print(json.dumps({"state": record["state"], "deletion_verified": record["deletion_verified"],
                              "estimated_cost_usd": record["estimated_cost_usd"], "training_exit": record.get("training_exit")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--max-hours", type=float, default=2.)
    a = ap.parse_args(); run(a.root, a.max_hours)
