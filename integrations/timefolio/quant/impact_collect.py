"""Record public Bybit spot messages, without credentials or order endpoints."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import websockets


async def collect(output: Path, symbol: str, seconds: int):
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    started = time.time()
    deadline = time.monotonic() + seconds
    segment = 0
    meta = {"source": "Bybit public spot WebSocket", "symbol": symbol,
            "depth": 50, "planned_seconds": seconds,
            "started_utc": datetime.now(timezone.utc).isoformat(), "errors": []}

    def save_status(complete=False):
        meta.update(counts=dict(counts), elapsed_seconds=round(time.time()-started, 2),
                    connections=segment, complete=complete)
        (output / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with gzip.open(output / "raw.jsonl.gz", "wt", encoding="utf-8") as raw:
        last_status = 0
        while time.monotonic() < deadline:
            try:
                async with websockets.connect("wss://stream.bybit.com/v5/public/spot",
                        ping_interval=None, open_timeout=15, max_size=8*1024*1024) as ws:
                    segment += 1
                    await ws.send(json.dumps({"op": "subscribe", "args": [
                        f"orderbook.50.{symbol}", f"publicTrade.{symbol}"]}))
                    last_ping = time.monotonic()
                    while time.monotonic() < deadline:
                        if time.monotonic()-last_ping >= 15:
                            await ws.send('{"op":"ping"}')
                            last_ping = time.monotonic()
                        try:
                            payload = await asyncio.wait_for(ws.recv(), timeout=min(5, max(.01, deadline-time.monotonic())))
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(payload)
                        raw.write(json.dumps({"segment": segment, "received_ns": time.time_ns(),
                                              "message": msg}, separators=(",", ":"))+"\n")
                        topic = msg.get("topic", msg.get("op", "other"))
                        counts[topic] += 1
                        if msg.get("success") is False:
                            raise RuntimeError(f"Subscription rejected: {msg.get('ret_msg')}")
                        if time.monotonic()-last_status >= 30:
                            raw.flush()
                            save_status()
                            print(json.dumps({"elapsed_s": meta["elapsed_seconds"], "counts": dict(counts)}), flush=True)
                            last_status = time.monotonic()
            except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
                meta["errors"].append({"type": type(exc).__name__, "elapsed_s": round(time.time()-started, 2)})
                save_status()
                await asyncio.sleep(min(2, max(0, deadline-time.monotonic())))
    meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save_status(complete=True)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 86400 or not args.symbol.isalnum():
        parser.error("Use an alphanumeric symbol and 10..86400 seconds")
    asyncio.run(collect(args.output, args.symbol.upper(), args.seconds))
