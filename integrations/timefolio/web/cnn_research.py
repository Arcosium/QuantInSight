"""Read-only results plus an automatic daily refresh of local paper accounts."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from fastapi import APIRouter

ROOT = Path(os.getenv("ARCTRADE_CNN_DIR", Path.home()/"vault"/"ArcTrade"/"krx_cnn"))
router = APIRouter(prefix="/api/krx-cnn")


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


@router.get("/status")
def status():
    current = _read(ROOT/"current.json")
    report = _read(Path(current["run_dir"])/current.get("report_file", "report.json")) if current.get("run_dir") else {}
    # Detailed backtest fills remain in the local artifact, not every UI poll.
    report["paths"] = {k: {"metrics": v["metrics"], "curve": v["curve"]}
                       for k, v in report.get("paths", {}).items()}
    paper = _read(ROOT/"paper.json")
    for account in paper.get("accounts", {}).values():
        book = account.get("book", {})
        account["trade_count"] = len(book.get("trades", []))
        account["positions"] = book.get("positions", {})
        account["cash"] = book.get("cash")
        account.pop("book", None)
        if account.get("pending"):
            account["pending"] = {k: v for k, v in account["pending"].items() if k not in ("previous", "adv")}
    signal = _read(ROOT/"signal.json")
    from quant.krx_cnn_data import universe
    name_map = universe()
    for row in signal.get("rows", []):
        row["name"] = name_map.get(row["code"], row["code"])
    return {"report": report, "signal": signal, "paper": paper,
            "worker": _read(ROOT/"worker.json")}


async def auto_loop():
    """Refresh before the next session; training is an explicit CLI action."""
    while True:
        proc = None
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            # Keep torch imports/CPU work outside the HTTP/event-loop process.
            with (ROOT/"worker.log").open("a") as log:
                proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "quant.krx_cnn_worker",
                    cwd=str(Path(__file__).resolve().parents[1]), stdout=log, stderr=log)
                await asyncio.wait_for(proc.wait(), timeout=600)
        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.terminate(); await proc.wait()
            raise
        except Exception:
            if proc and proc.returncode is None:
                proc.terminate(); await proc.wait()
        await asyncio.sleep(1800)
