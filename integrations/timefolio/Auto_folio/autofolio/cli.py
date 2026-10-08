from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .engine import AutoFolioEngine
from .timefolio_browser import TimefolioBrowser, TimefolioCredentials


def _load_json_arg(value: str):
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto_folio KR mock trading planner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Build QuantInSight-style KR mock orders")
    plan.add_argument("--targets", required=True, help='JSON list or file, e.g. ["005930"]')
    plan.add_argument("--holdings", default="[]", help="JSON list or file")
    plan.add_argument("--buying-power", required=True, help='JSON object or file: {"cash":...,"total_eval":...,"ok":true}')
    plan.add_argument("--prices", required=True, help='JSON object or file: {"005930":70000}')
    plan.add_argument("--session", default=None)
    plan.add_argument("--data-dir", default="data")

    login = sub.add_parser("login-check", help="Verify Timefolio login with Playwright")
    login.add_argument("--username", default=os.getenv("TIMEFOLIO_USERNAME", ""))
    login.add_argument("--password", default=os.getenv("TIMEFOLIO_PASSWORD", ""))
    login.add_argument("--headed", action="store_true")

    args = parser.parse_args()
    if args.cmd == "plan":
        engine = AutoFolioEngine(data_dir=args.data_dir)
        out = engine.plan_cycle(
            target_codes=_load_json_arg(args.targets),
            holdings=_load_json_arg(args.holdings),
            buying_power=_load_json_arg(args.buying_power),
            price_map=_load_json_arg(args.prices),
            session=args.session,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "login-check":
        if not args.username or not args.password:
            raise SystemExit("TIMEFOLIO_USERNAME/TIMEFOLIO_PASSWORD or --username/--password required")
        with TimefolioBrowser(headless=not args.headed) as browser:
            status = browser.login(TimefolioCredentials(args.username, args.password))
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
