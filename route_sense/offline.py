"""Learn from logs: per-model reports, significance tests, and estimates of how a different policy would have done.

Two estimators, and the assumptions behind each are the whole point:

* ``replay`` needs a log where the model for each call was chosen **uniformly at random** (a short randomised
  rollout is how you get one). It is then unbiased.
* ``ips`` works on any log that recorded the probability of each choice (``propensity``), including logs from an
  adaptive router, and divides by the sum of weights (self-normalised) so a few huge weights do not dominate.

Neither can say anything about a model that never appears in the log.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .router import LogRecord, Router
from .stats import Welford, two_sided_p


def to_jsonl(records: Iterable[LogRecord]) -> str:
    return "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in records)


def from_jsonl(text: str) -> list[LogRecord]:
    out = []
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(LogRecord(d["arm"], float(d["quality"]), float(d.get("cost", 0.0)), d.get("latency"), d.get("context"),
                                 d.get("propensity"), bool(d.get("ok", True))))
    return out


@dataclass(frozen=True)
class ArmReport:
    arm: str
    n: int
    quality: float
    cost: float
    reward: float
    reward_low: float  # 95% normal-approximation interval on mean reward
    reward_high: float
    failures: int


def arm_report(records: Iterable[LogRecord], cost_weight: float = 0.0) -> dict[str, ArmReport]:
    acc: dict[str, tuple[Welford, Welford, Welford, int]] = {}
    for r in records:
        q, c, rw, f = acc.setdefault(r.arm, (Welford(), Welford(), Welford(), 0))
        q.add(r.quality)
        c.add(r.cost)
        rw.add(r.quality - cost_weight * r.cost)
        acc[r.arm] = (q, c, rw, f + (0 if r.ok else 1))
    out = {}
    for arm, (q, c, rw, f) in acc.items():
        half = 1.96 * rw.sem if rw.n > 1 else math.inf
        out[arm] = ArmReport(arm, rw.n, q.mean, c.mean, rw.mean, rw.mean - half, rw.mean + half, f)
    return out


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    reward_diff: float  # b minus a
    quality_diff: float
    cost_diff: float
    z: float
    p_value: float
    significant: bool
    per_1000_cost: float  # cost change per 1000 calls if traffic moved from a to b
    per_1000_quality: float  # quality points per 1000 calls


def compare(records: list[LogRecord], a: str, b: str, cost_weight: float = 0.0, alpha: float = 0.05) -> Comparison:
    """Welch-style z test of mean net reward, b versus a. With fewer than two calls on either side it is not significant."""
    ra, rb = [r for r in records if r.arm == a], [r for r in records if r.arm == b]
    if not ra or not rb:
        raise ValueError("both arms need at least one record")
    wa, wb, qa, qb, ca, cb = Welford(), Welford(), Welford(), Welford(), Welford(), Welford()
    for r in ra:
        wa.add(r.quality - cost_weight * r.cost)
        qa.add(r.quality)
        ca.add(r.cost)
    for r in rb:
        wb.add(r.quality - cost_weight * r.cost)
        qb.add(r.quality)
        cb.add(r.cost)
    diff = wb.mean - wa.mean
    se = math.sqrt((wa.var / wa.n) + (wb.var / wb.n)) if wa.n > 1 and wb.n > 1 else math.inf
    z = 0.0 if math.isinf(se) else (diff / se if se > 0 else (math.inf if diff != 0 else 0.0))
    p = 1.0 if math.isinf(se) else (two_sided_p(z) if math.isfinite(z) else 0.0)
    return Comparison(a, b, diff, qb.mean - qa.mean, cb.mean - ca.mean, z, p, p < alpha, (cb.mean - ca.mean) * 1000, (qb.mean - qa.mean) * 1000)


@dataclass(frozen=True)
class ReplayResult:
    matched: int
    total: int
    mean_reward: float
    mean_quality: float
    mean_cost: float


def replay(records: list[LogRecord], make_router: Callable[[], Router]) -> ReplayResult:
    """Run a fresh router over the log. It learns only from calls where the logged model is the one it would have
    picked, and the estimate is the mean over those matches. Unbiased only if the log's choices were uniform random."""
    router = make_router()
    reward, quality, cost, matched = 0.0, 0.0, 0.0, 0
    for r in records:
        if r.arm not in router.stats:
            continue
        if router.select() == r.arm:
            router.update(r.arm, r.quality, r.cost, latency=r.latency, ok=r.ok)
            reward += r.quality - router.cost_weight * r.cost
            quality += r.quality
            cost += r.cost
            matched += 1
    n = max(matched, 1)
    return ReplayResult(matched, len(records), reward / n, quality / n, cost / n)


@dataclass(frozen=True)
class IpsResult:
    value: float  # estimated mean net reward under the target policy
    effective_sample_size: float
    max_weight: float


def ips(records: list[LogRecord], target: Callable[[LogRecord], dict[str, float]], cost_weight: float = 0.0, clip: float = 20.0) -> IpsResult:
    """Self-normalised inverse propensity estimate of a target policy's mean net reward.

    ``target(record)`` gives the target policy's probability for each arm in that situation. Weights are clipped at
    ``clip`` to bound variance, which adds a little bias in exchange. Every record needs a positive ``propensity``.
    """
    num = den = den2 = peak = 0.0
    for r in records:
        if r.propensity is None or r.propensity <= 0:
            raise ValueError("every record needs a positive propensity")
        w = min(target(r).get(r.arm, 0.0) / r.propensity, clip)
        num += w * (r.quality - cost_weight * r.cost)
        den += w
        den2 += w * w
        peak = max(peak, w)
    if den == 0:
        return IpsResult(0.0, 0.0, 0.0)
    return IpsResult(num / den, den * den / den2, peak)
