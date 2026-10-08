"""One bounded refresh cycle; serialized across restarts/manual invocations."""
import fcntl
import json
import time
from pathlib import Path
import pandas as pd
from quant.krx_cnn_data import DAILY, US_DAILY, atomic_json, refresh, refresh_us_policy
from quant.krx_cnn import ARTIFACTS, latest_signal
from quant.krx_cnn_paper import update


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS/"worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            status_path = DAILY/"status.json"
            now = pd.Timestamp.now(tz="Asia/Seoul")
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            updated = pd.Timestamp(status.get("updated_at", "2000-01-01T00:00:00Z")).tz_convert("Asia/Seoul")
            # Full snapshot once a day, including failures; no intraday candle use.
            if updated.date() != now.date():
                status = refresh()
            us_path = US_DAILY/"status.json"
            us_status = json.loads(us_path.read_text()) if us_path.exists() else {}
            us_updated = pd.Timestamp(us_status.get("updated_at", "2000-01-01T00:00:00Z"))
            if (pd.Timestamp.now(tz="UTC")-us_updated).total_seconds() > 6*3600:
                us_status = refresh_us_policy()
            if (ARTIFACTS/"current.json").exists():
                signal = latest_signal()
                atomic_json(ARTIFACTS/"signal.json", signal)
                update(ARTIFACTS, signal)
            atomic_json(ARTIFACTS/"worker.json", {"status": "ok", "updated_at": now.isoformat(),
                        "data_as_of": status.get("as_of"), "data_ok": status.get("ok"),
                        "us_as_of": us_status.get("as_of"), "us_ok": us_status.get("ok")})
        except Exception as e:
            atomic_json(ARTIFACTS/"worker.json", {"status": "error", "error": type(e).__name__,
                        "updated_at": pd.Timestamp.now(tz="UTC").isoformat()})
            raise


if __name__ == "__main__":
    run()
