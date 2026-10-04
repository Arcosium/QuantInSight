"""Paired within-model interventions in the closed investor society.

python3 -m research.society_shocks --people 1000 --warmup 60 --horizon 60
Optional: --hybrid-validation (local LLM smoke comparison in the first world).

At the same checkpoint, the same target residents receive changes in their
conformity, on its bounded 0..1 scale. Temporary changes are restored without
undoing the resulting trades, memories or beliefs. No money is injected.

Random streams are keyed by future seed, day, phase, resident, asset and source
call site. Conditional draws cannot advance unrelated actors' streams. The
economic random tape is reconciled in every matched pair. Source digests are
recorded because source call sites are part of this coupling scheme.

This is a native prototype, not MiroFish. All reported effects are causal
contrasts INSIDE this particular uncalibrated model, not empirical effects.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from research.investor_society import ROOT, VERSION as MODEL_VERSION, LocalBeliefs, Society, save_json

VERSION = "society-shocks-v1-addressed-randomness"
SEEDS = (7, 42, 99, 2026, 13, 21, 55, 89, 144, 233, 377, 610)
CASES = (
    {"name": "sham", "delta": 0.0, "duration": None},
    {"name": "pulse_up", "delta": .2, "duration": 1},
    {"name": "twenty_days_up", "delta": .2, "duration": 20},
    {"name": "persistent_up", "delta": .2, "duration": None},
    {"name": "persistent_down", "delta": -.2, "duration": None},
    {"name": "persistent_small_up", "delta": .1, "duration": None},
    {"name": "persistent_large_up", "delta": .4, "duration": None},
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class AddressedRandom:
    """Expose Random operations with independent per-call-site streams."""
    def __init__(self, seed: int, state):
        self.seed = seed
        self.context = ("initial",)
        self.streams = {}
        self.sentinel = random.Random()
        self.sentinel.setstate(state)
        self.trace = None

    def getstate(self):
        # Core checkpoint schema compatibility only. Future draws are addressed
        # by absolute day, not this sentinel. Runner checkpoints carry metadata.
        return self.sentinel.getstate()

    def select(self, *context):
        self.context, self.streams = context, {}

    @contextmanager
    def scope(self, *context, trace=False):
        previous = self.context, self.streams, self.trace
        self.select(*context)
        self.trace = hashlib.sha256() if trace else None
        try:
            yield self
        finally:
            self.context, self.streams, self.trace = previous

    def invoke(self, name, args, kwargs):
        caller = sys._getframe(2)
        site = (Path(caller.f_code.co_filename).name, caller.f_lineno, name)
        if site not in self.streams:
            seed = int(digest([self.seed, *self.context, *site])[:16], 16)
            self.streams[site] = random.Random(seed)
        result = getattr(self.streams[site], name)(*args, **kwargs)
        if self.trace is not None:
            # Shuffle identity, never a resident's evolving balance sheet.
            trace_result = [getattr(x, "id", x) for x in args[0]] if name == "shuffle" else result
            self.trace.update(json.dumps([site, trace_result], separators=(",", ":")).encode())
        return result

    def __getattr__(self, name):
        if name not in {"random", "randint", "randrange", "gauss", "choice", "choices", "shuffle"}:
            raise AttributeError(name)
        def operation(*args, **kwargs):
            return self.invoke(name, args, kwargs)
        return operation


class CoupledSociety(Society):
    @classmethod
    def from_checkpoint(cls, checkpoint: dict, future_seed: int):
        obj = Society.restore(copy.deepcopy(checkpoint))
        obj.__class__ = cls
        obj.rng = AddressedRandom(future_seed, obj.rng.getstate())
        obj.in_orders = False
        obj.economic_tape = ""
        return obj

    def economy(self):
        with self.rng.scope(self.day, "economy", trace=True):
            result = super().economy()
            self.economic_tape = self.rng.trace.hexdigest()
        return result

    def orders(self):
        self.in_orders = True
        try:
            with self.rng.scope(self.day, "orders"):
                return super().orders()
        finally:
            self.in_orders = False

    def observation(self, a):
        result = super().observation(a)
        if self.in_orders:
            self.rng.select(self.day, "resident_orders", a.id)
        return result

    def signal(self, a, j, obs):
        with self.rng.scope(self.day, "signal", a.id, j):
            return super().signal(a, j, obs)


def choose_targets(checkpoint: dict, fraction: float, seed: int, selection="random") -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = len(checkpoint["residents"])
    count = max(1, round(n * fraction))
    if selection == "random":
        rng = random.Random(int(digest([seed, "treatment_cohort"])[:16], 16))
        return sorted(rng.sample(range(n), count))
    if selection == "wealth":
        def wealth(a):
            return a["cash"] + sum(q * f["price"] for q, f in zip(a["shares"], checkpoint["firms"])) - a["debt"] - a["interest_due"]
        return sorted(a["id"] for a in sorted(checkpoint["residents"], key=wealth, reverse=True)[:count])
    raise ValueError("Unknown target selection")


def apply_conformity(society: Society, targets: list[int], originals: dict[int, float], delta: float):
    if not math.isfinite(delta) or not -1 <= delta <= 1 or len(targets) != len(set(targets)):
        raise ValueError("Invalid intervention")
    for i in targets:
        if i not in originals or not 0 <= i < len(society.residents):
            raise ValueError("Invalid resident target")
        society.residents[i].conformity = max(0.0, min(1.0, originals[i] + delta))


def group_state(society: Society, targets: set[int]) -> dict:
    groups = {"target": {"shares": [0, 0, 0], "wealth_cents": 0, "cash_cents": 0, "debt_cents": 0},
              "other": {"shares": [0, 0, 0], "wealth_cents": 0, "cash_cents": 0, "debt_cents": 0}}
    for a in society.residents:
        g = groups["target" if a.id in targets else "other"]
        for j, q in enumerate(a.shares):
            g["shares"][j] += q
        g["wealth_cents"] += society.wealth(a)
        g["cash_cents"] += a.cash
        g["debt_cents"] += a.debt + a.interest_due
    return groups


def run_branch(checkpoint, future_seed, targets, delta, duration, horizon, llm=None):
    if horizon < 1 or duration is not None and duration < 1:
        raise ValueError("Invalid time horizon")
    if not targets or len(targets) != len(set(targets)) or any(not 0 <= i < len(checkpoint['residents']) for i in targets):
        raise ValueError("Invalid resident targets")
    s = CoupledSociety.from_checkpoint(checkpoint, future_seed)
    originals = {i: s.residents[i].conformity for i in targets}
    apply_conformity(s, targets, originals, delta)
    changed = s.checkpoint()
    for i in targets:
        changed["residents"][i]["conformity"] = originals[i]
    if digest(changed) != digest(checkpoint):
        raise AssertionError("Intervention changed something besides conformity")
    realized_delta = statistics.fmean(s.residents[i].conformity - originals[i] for i in targets)
    membership = set(targets)
    start_groups = group_state(s, membership)
    start_price = checkpoint["prices"][-1]
    path = [{"offset": 0, "day": s.day, "index": 1.0, "prices_cents": start_price,
             "groups": start_groups, "effective_delta": realized_delta}]
    for h in range(1, horizon + 1):
        if duration is not None and h > duration:
            apply_conformity(s, targets, originals, 0)
        first_trade, first_order = len(s.transactions), len(s.decisions)
        daily = s.step(llm)
        trades = s.transactions[first_trade:]
        decisions = s.decisions[first_order:]
        requested = sum(o["quantity"] for o in decisions)
        net_value = sum(t["shares"] * t["price_cents"] * ((t["buyer"] in membership) - (t["seller"] in membership)) for t in trades)
        path.append({"offset": h, "day": s.day,
                     "index": statistics.fmean(p / op for p, op in zip(daily["prices_cents"], start_price)),
                     "prices_cents": daily["prices_cents"], "trades": daily["trades"],
                     "volume_shares": daily["volume_shares"], "submitted_orders": len(decisions),
                     "submitted_shares_two_sided": requested,
                     "filled_shares_two_sided": 2 * daily["volume_shares"],
                     "fill_rate": 2 * daily["volume_shares"] / requested if requested else None,
                     "target_net_buy_value_cents": net_value,
                     "consumption_shortages": daily["consumption_shortages"],
                     "consumption_cents": daily["consumption_cents"],
                     "no_wage_residents": daily["no_wage_residents"],
                     "household_debt_cents": daily["household_debt_cents"],
                     "groups": group_state(s, membership),
                     "economic_tape_sha256": s.economic_tape,
                     "effective_delta": statistics.fmean(s.residents[i].conformity - originals[i] for i in targets),
                     "cash_error": daily["cash_conservation_error"],
                     "aggregate_net_shares": daily["aggregate_resident_net_shares"]})
    return {"path": path, "target_ids": targets, "nominal_delta": delta,
            "effective_initial_delta": realized_delta, "duration_days": duration,
            "target_wealth_fraction_at_shock": start_groups["target"]["wealth_cents"] / sum(g["wealth_cents"] for g in start_groups.values()),
            "initial_state_equal_except_trait": True,
            "final_conformity": [s.residents[i].conformity for i in targets],
            "llm_audit": s.llm_audit[len(checkpoint["llm_audit"]):]}


def ratio_change(new, old):
    return 100 * (new / old - 1) if old > 0 else None


def contrast(control: dict, treatment: dict) -> dict:
    if len(control["path"]) != len(treatment["path"]):
        raise ValueError("Unmatched horizons")
    effects = []
    initial = control["path"][0]["groups"]["target"]["shares"]
    for c, t in zip(control["path"][1:], treatment["path"][1:]):
        if c["economic_tape_sha256"] != t["economic_tape_sha256"] or c["day"] != t["day"]:
            raise AssertionError("Exogenous economic events differ between branches")
        for branch in (c, t):
            if branch["cash_error"] or any(branch["aggregate_net_shares"]):
                raise AssertionError("Economic conservation failed")
            if branch["filled_shares_two_sided"] > branch["submitted_shares_two_sided"]:
                raise AssertionError("Fill count exceeds order inventory")
        group_net = [a - b for a, b in zip(t["groups"]["target"]["shares"], c["groups"]["target"]["shares"])]
        other_net = [a - b for a, b in zip(t["groups"]["other"]["shares"], c["groups"]["other"]["shares"])]
        if any(a + b for a, b in zip(group_net, other_net)):
            raise AssertionError("Group net-share effects do not cancel")
        effects.append({"offset": c["offset"], "price_gap_pct": ratio_change(t["index"], c["index"]),
                        "volume_difference": t["volume_shares"] - c["volume_shares"],
                        "trades_difference": t["trades"] - c["trades"],
                        "fill_rate_difference_pp": 100 * (t["fill_rate"] - c["fill_rate"]) if c["fill_rate"] is not None and t["fill_rate"] is not None else None,
                        "target_net_shares_effect": group_net, "other_net_shares_effect": other_net,
                        "target_cumulative_net_shares": [q - op for q, op in zip(t["groups"]["target"]["shares"], initial)],
                        "target_wealth_gap_pct": ratio_change(t["groups"]["target"]["wealth_cents"], c["groups"]["target"]["wealth_cents"]),
                        "other_wealth_gap_pct": ratio_change(t["groups"]["other"]["wealth_cents"], c["groups"]["other"]["wealth_cents"]),
                        "debt_difference_cents": t["household_debt_cents"] - c["household_debt_cents"],
                        "consumption_difference_cents": t["consumption_cents"] - c["consumption_cents"],
                        "shortage_difference_people": t["consumption_shortages"] - c["consumption_shortages"],
                        "no_wage_difference_people": t["no_wage_residents"] - c["no_wage_residents"]})
    gaps = [e["price_gap_pct"] for e in effects]
    peak = max(range(len(gaps)), key=lambda i: abs(gaps[i]))
    # A stable five-day return inside 0.5 percentage points after the peak.
    recovery = next((i + 1 for i in range(peak + 1, len(gaps) - 4)
                     if all(abs(x) <= .5 for x in gaps[i:i + 5])), None)
    def volatility(path):
        return statistics.pstdev(b["index"] / a["index"] - 1 for a, b in zip(path, path[1:])) * 100
    cvol, tvol = volatility(control["path"]), volatility(treatment["path"])
    cvolume = sum(x["volume_shares"] for x in control["path"][1:])
    tvolume = sum(x["volume_shares"] for x in treatment["path"][1:])
    def fill(path):
        wanted = sum(x["submitted_shares_two_sided"] for x in path[1:])
        return sum(x["filled_shares_two_sided"] for x in path[1:]) / wanted if wanted else None
    cf, tf = fill(control["path"]), fill(treatment["path"])
    return {"effects": effects, "metrics": {
        "peak_price_gap_pct": gaps[peak], "peak_abs_price_gap_pct": abs(gaps[peak]),
        "peak_offset": peak + 1, "terminal_price_gap_pct": gaps[-1],
        "mean_abs_price_gap_pct": statistics.fmean(abs(x) for x in gaps),
        "integrated_abs_gap_pct_days": sum(abs(x) for x in gaps),
        "recovery_offset_inside_half_pct_for_5_days": recovery,
        "recovery_censored": recovery is None,
        "volume_change_pct": ratio_change(tvolume, cvolume),
        "control_daily_volatility_pct": cvol, "treatment_daily_volatility_pct": tvol,
        "volatility_difference_pp": tvol - cvol,
        "fill_rate_difference_pp": 100 * (tf - cf) if cf is not None and tf is not None else None,
        "terminal_target_wealth_gap_pct": effects[-1]["target_wealth_gap_pct"],
        "terminal_other_wealth_gap_pct": effects[-1]["other_wealth_gap_pct"],
        "terminal_debt_difference_cents": effects[-1]["debt_difference_cents"],
        "cumulative_shortage_difference_person_days": sum(x["shortage_difference_people"] for x in effects),
        "economic_tapes_equal": True}}


def distribution(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "positive_fraction": None}
    def quantile(p):
        position = (len(values) - 1) * p
        a, b = math.floor(position), math.ceil(position)
        return values[a] + (values[b] - values[a]) * (position - a)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p10": quantile(.1), "p90": quantile(.9),
            "positive_fraction": sum(x > 0 for x in values) / len(values)}


def aggregate(results):
    output = {}
    for case in sorted({r["case"] for r in results}):
        rows = [r for r in results if r["case"] == case]
        keys = [k for k, v in rows[0]["metrics"].items() if k not in {"economic_tapes_equal", "recovery_censored"} and isinstance(v, (int, float, type(None)))]
        output[case] = {"worlds": len(rows),
                        "metrics": {key: distribution([r["metrics"][key] for r in rows]) for key in keys},
                        "mean_effective_delta": statistics.fmean(r["effective_initial_delta"] for r in rows),
                        "recovery_censored_worlds": sum(r["metrics"]["recovery_censored"] for r in rows),
                        "path": [{"offset": h + 1,
                                  **{key: distribution([r["effects"][h][key] for r in rows]) for key in ("price_gap_pct", "volume_difference", "debt_difference_cents", "shortage_difference_people")}}
                                 for h in range(len(rows[0]["effects"]))]}
    return output


def run_world(run: Path, seed: int, people: int, warmup: int, horizon: int,
              fraction: float, selection: str) -> list[dict]:
    """Independent world; writes only its own seed subdirectory."""
    initial = Society(people, seed).checkpoint()
    warm = CoupledSociety.from_checkpoint(initial, seed)
    for _ in range(warmup):
        warm.step()
    checkpoint = warm.checkpoint()
    targets = choose_targets(checkpoint, fraction, seed, selection)
    directory = run / str(seed)
    directory.mkdir()
    save_json(directory / "checkpoint.json", {"society": checkpoint, "coupling_seed": seed, "resume_with": "research.society_shocks.CoupledSociety.from_checkpoint"})
    save_json(directory / "targets.json", targets)
    control = run_branch(checkpoint, seed, targets, 0, None, horizon)
    save_json(directory / "control.json", control)
    results = []
    for case in CASES:
        branch = run_branch(checkpoint, seed, targets, case["delta"], case["duration"], horizon)
        effects = contrast(control, branch)
        if case["name"] == "sham" and branch != control:
            raise AssertionError("Zero intervention is not an exact replay")
        save_json(directory / (case["name"] + ".json"), branch)
        results.append({"seed": seed, "case": case["name"], "nominal_delta": case["delta"],
                        "effective_initial_delta": branch["effective_initial_delta"],
                        "target_wealth_fraction": branch["target_wealth_fraction_at_shock"], **effects})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=.2)
    parser.add_argument("--selection", choices=("random", "wealth"), default="random")
    parser.add_argument("--hybrid-validation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.worlds <= len(SEEDS) or not 1 <= args.workers <= 4 or args.warmup < 0 or args.horizon < 5:
        parser.error("Invalid world count or horizon")
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    run = ROOT / "data" / "society_shocks" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    hashes = {name: hashlib.sha256((ROOT / "research" / name).read_bytes()).hexdigest()
              for name in ("society_shocks.py", "investor_society.py")}
    manifest = {"version": VERSION, "model_version": MODEL_VERSION, "source_sha256": hashes,
                "people": args.people, "warmup_days": args.warmup, "horizon_days": args.horizon,
                "world_seeds": SEEDS[:args.worlds], "target_fraction": args.fraction,
                "workers": args.workers,
                "selection": args.selection, "cases": CASES,
                "primary_cognition": "rules; no new LLM calls", "index": "equal-weight three-sector price relatives, reset to 1 at checkpoint",
                "uncertainty": "p10..p90 across synthetic worlds, not a confidence interval for a real-market effect",
                "random_coupling": "by future seed/day/phase/resident/asset/source call site",
                "limitations": ["uncalibrated native model, not MiroFish or observed humans",
                                "initial profiles and histories vary across worlds; this does not isolate future-noise uncertainty conditional on one history",
                                "closed resident-only market: aggregate personal net shares are zero",
                                "fixed wages and goods prices limit real-economy transmission",
                                "daily expiring order book; fill rate is not a real-world spread/depth measure",
                                "group wealth effects include income and consumption; not trading alpha"]}
    save_json(run / "manifest.json", manifest)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_world, run, seed, args.people, args.warmup,
                                   args.horizon, args.fraction, args.selection): seed
                   for seed in SEEDS[:args.worlds]}
        for future in as_completed(futures):
            results.extend(future.result())
            results.sort(key=lambda r: (SEEDS.index(r['seed']), r['case']))
            save_json(run / "results.json", results)
            print(json.dumps({"world_seed": futures[future], "completed_worlds": len(results) // len(CASES), "run": str(run)}), flush=True)
    report = {"version": VERSION, "verified_real_market_alpha": False,
              "worlds": args.worlds, "paired_cases": len(results),
              "cases": aggregate(results), "limitations": manifest["limitations"]}
    save_json(run / "report.json", report)
    if args.hybrid_validation:
        first_checkpoint = json.loads((run / str(SEEDS[0]) / "checkpoint.json").read_text())["society"]
        first_targets = json.loads((run / str(SEEDS[0]) / "targets.json").read_text())
        from config import LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL
        cache = {}
        requester = LocalBeliefs(LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL)
        def frozen_transport(messages):
            key = digest(messages)
            if key not in cache:
                cache[key] = requester.request(messages)
            return copy.deepcopy(cache[key])
        def cognition():
            return LocalBeliefs(LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL, agents=4, every=15,
                               workers=1, max_calls=8, transport=frozen_transport)
        seed = SEEDS[0]
        baseline = run_branch(first_checkpoint, seed, first_targets, 0, None, min(20, args.horizon), cognition())
        altered = run_branch(first_checkpoint, seed, first_targets, .2, None, min(20, args.horizon), cognition())
        hybrid = {"purpose": "hybrid smoke comparison; distinct LLM inputs can have independent response noise; not primary effect estimation",
                  "unique_model_requests": len(cache), "comparison": contrast(baseline, altered),
                  "control": baseline, "treatment": altered}
        save_json(run / "hybrid_validation.json", hybrid)
        save_json(run / "frozen_llm_responses.json", cache)
    print(json.dumps({"run": str(run), "worlds": args.worlds, "paired_cases": len(results)}))


if __name__ == "__main__":
    main()
