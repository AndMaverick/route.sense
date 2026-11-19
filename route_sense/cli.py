"""``python -m route_sense``: read a call log (JSONL) and report. Local only."""
from __future__ import annotations

import argparse
import json
import sys

from . import offline
from .policies import UCB1, CheapestAboveFloor, EpsilonGreedy, Policy, Thompson
from .router import Router


def _policy(name: str, floor: float | None) -> Policy:
    if name == "thompson":
        return Thompson()
    if name == "epsilon":
        return EpsilonGreedy()
    if name == "floor":
        if floor is None:
            raise SystemExit("--floor is required for the floor policy")
        return CheapestAboveFloor(floor)
    return UCB1()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="route_sense", description="Report on a model-call log and test routing policies against it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="per-model quality, cost, net reward and significance versus a baseline")
    rep.add_argument("file")
    rep.add_argument("--cost-weight", type=float, default=0.0)
    rep.add_argument("--baseline", help="arm to compare every other arm against")
    rep.add_argument("--json", action="store_true")
    rp = sub.add_parser("replay", help="estimate a policy's value on a log whose choices were uniformly random")
    rp.add_argument("file")
    rp.add_argument("--policy", choices=["ucb", "thompson", "epsilon", "floor"], default="ucb")
    rp.add_argument("--cost-weight", type=float, default=0.0)
    rp.add_argument("--floor", type=float)
    args = ap.parse_args(argv)
    with open(args.file, encoding="utf-8") as fh:
        records = offline.from_jsonl(fh.read())
    if not records:
        print("error: the log is empty", file=sys.stderr)
        return 2
    if args.cmd == "report":
        reports = offline.arm_report(records, args.cost_weight)
        comps = []
        if args.baseline:
            if args.baseline not in reports:
                print(f"error: no records for baseline {args.baseline}", file=sys.stderr)
                return 2
            comps = [offline.compare(records, args.baseline, a, args.cost_weight) for a in reports if a != args.baseline]
        if args.json:
            json.dump({"arms": {a: vars(r) for a, r in reports.items()}, "comparisons": [vars(c) for c in comps]}, sys.stdout, indent=2, default=str)
            print()
            return 0
        print(f"{'arm':<20} {'n':>6} {'quality':>8} {'cost':>9} {'net reward':>11} {'95% interval':>20} {'fail':>5}")
        for r in sorted(reports.values(), key=lambda x: -x.reward):
            print(f"{r.arm:<20} {r.n:>6} {r.quality:>8.3f} {r.cost:>9.4f} {r.reward:>11.4f} {f'[{r.reward_low:.3f}, {r.reward_high:.3f}]':>20} {r.failures:>5}")
        for c in comps:
            verdict = "significant" if c.significant else "not significant"
            print(f"\n{c.b} vs {c.a}: net reward {c.reward_diff:+.4f}, quality {c.quality_diff:+.4f}, cost {c.cost_diff:+.4f} "
                  f"(per 1000 calls: cost {c.per_1000_cost:+.2f}), p = {c.p_value:.3f}, {verdict}")
        return 0
    arms = list(dict.fromkeys(r.arm for r in records))
    res = offline.replay(records, lambda: Router(arms, cost_weight=args.cost_weight, policy=_policy(args.policy, args.floor)))
    print(f"matched {res.matched} of {res.total} calls; estimated net reward {res.mean_reward:.4f}, quality {res.mean_quality:.4f}, cost {res.mean_cost:.4f}")
    print("valid only if the logged model for each call was chosen uniformly at random")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
