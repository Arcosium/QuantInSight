"""Offline chronological replay of the new stock admission rule.

This measures the deterministic research rule only, not the full committee,
broker fills or historical account returns. Uses the current stored universe,
so survivorship and cached-data coverage limit interpretation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.empirical_edge import estimate_edge
from tools.market_data import load_daily_csv


def replay(codes, periods=24, threshold=.8):
    histories = {}
    for code in codes:
        df = load_daily_csv(code)
        if df is not None and len(df) >= 180:
            histories[code] = df.drop_duplicates("date",keep="last").sort_values("date").reset_index(drop=True)
    dates = sorted({pd.Timestamp(x) for df in histories.values() for x in df.date})
    dates = dates[-(periods*5+6):]
    windows = []
    for i in range(0,len(dates)-5,5):
        signal, exit_date = dates[i], dates[i+5]
        candidates, benchmark = [], []
        future = {}
        signal_names = []
        for code,df in histories.items():
            positions = df.index[df.date == signal].tolist()
            if not positions:
                continue
            pos = positions[-1]
            # Freeze eligibility before looking at any future prices or coverage.
            ev = estimate_edge(df.iloc[:pos+1], cost_pct=.35,min_net_edge_pct=threshold,
                               as_of=signal+pd.Timedelta(days=1))
            signal_names.append(code)
            if ev.eligible:
                candidates.append((ev.conservative_net_pct,code))
        picked = sorted(candidates,reverse=True)[:5]
        selected = [r[1] for r in picked]
        for code in signal_names:
            df = histories[code]
            fwd = df[(df.date > signal) & (df.date <= exit_date)]
            if len(fwd) != 5 or pd.Timestamp(fwd.iloc[-1].date) != exit_date:
                continue
            entry, close = float(fwd.iloc[0].open), float(fwd.iloc[-1].close)
            if not np.isfinite(entry+close) or entry <= 0 or close <= 0:
                continue
            forward = (close/entry-1)*100-.35
            future[code] = forward
            benchmark.append(forward)
        missing = [c for c in selected if c not in future]
        coverage = len(benchmark)/len(signal_names) if signal_names else 0
        valid = not missing and coverage >= .95
        windows.append({"signal_date":str(signal.date()),"exit_date":str(exit_date.date()),
                        "selected":selected,"eligible":len(candidates),
                        "net_return_pct":float(np.mean([future[c] for c in selected])) if selected and not missing else (0 if not selected else None),
                        "universe_return_pct":float(np.mean(benchmark)) if benchmark else None,
                        "benchmark_names":len(benchmark),"forward_coverage":coverage,
                        "missing_selected":missing,"valid":valid})
    def metrics(key):
        if not windows or any(not w["valid"] or w[key] is None for w in windows):
            return {"return_pct":None,"five_day_drawdown_pct":None,"reason":"incomplete_forward_coverage"}
        returns=[w[key]/100 for w in windows]
        nav=np.r_[1,np.cumprod(1+np.asarray(returns))]
        return {"return_pct":float((nav[-1]-1)*100),
                "five_day_drawdown_pct":float((nav/np.maximum.accumulate(nav)-1).min()*100)}
    return {"scope":"stock admission rule only; no committee or broker simulation",
            "limitations":["current universe survivorship bias","cached prices, not point-in-time vendor data",
                           "five-day snapshots understate intraperiod drawdown", "cost is an assumption"],
            "promotion_status":"not_validated",
            "valid_windows":sum(w["valid"] for w in windows),
            "threshold_pct":threshold,"round_trip_cost_pct":.35,"histories":len(histories),
            "rule":metrics("net_return_pct"),"equal_weight_universe":metrics("universe_return_pct"),
            "cash_windows":sum(not w["selected"] for w in windows),"windows":windows}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    universe=Path(__file__).resolve().parents[1]/"data/universe.csv"
    with universe.open() as f:
        codes=[r["code"] for r in csv.DictReader(f)]
    report=replay(codes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "windows"},ensure_ascii=False))
