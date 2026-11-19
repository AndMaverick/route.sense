"""The router: learns from outcomes which model earns its cost, within the limits you set."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .policies import UCB1, Policy, policy_from_dict
from .stats import Welford


class NoEligibleArm(RuntimeError):
    """Every arm was excluded by a tag, cost, latency, budget or circuit-breaker rule, and no fallback is set."""


@dataclass(frozen=True)
class ArmSpec:
    """A model and what is known about it before any traffic. ``cost_hint`` and ``latency_hint`` are used for
    filtering until real observations replace them."""

    name: str
    tags: frozenset[str] = frozenset()
    cost_hint: float | None = None
    latency_hint: float | None = None


@dataclass
class Budget:
    limit: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


@dataclass
class ArmStats:
    pulls: int = 0
    dn: float = 0.0  # discounted pull count
    dsum: float = 0.0  # discounted reward sum
    total: float = 0.0
    reward: Welford = field(default_factory=Welford)
    quality: Welford = field(default_factory=Welford)
    cost: Welford = field(default_factory=Welford)
    latency: Welford = field(default_factory=Welford)
    failures: int = 0
    consecutive_failures: int = 0
    open_until: int | None = None


@dataclass(frozen=True)
class Decision:
    selection: int
    arm: str
    candidates: tuple[str, ...]
    propensity: float | None


@dataclass(frozen=True)
class LogRecord:
    """One routed call and its outcome, in the shape the offline tools read."""

    arm: str
    quality: float
    cost: float = 0.0
    latency: float | None = None
    context: str | None = None
    propensity: float | None = None
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "quality": self.quality, "cost": self.cost, "latency": self.latency, "context": self.context,
                "propensity": self.propensity, "ok": self.ok}


class Router:
    """Pick among arms (models) by learned net reward: quality minus ``cost_weight`` times cost.

    Every arm is tried once, in order, then the policy (UCB1 by default) decides. Around that core:

    * ``discount`` below 1 forgets old observations, so a model that gets worse (or better) is noticed.
    * ``trip_after`` consecutive failures open a circuit breaker on an arm for ``cooldown`` selections, after which
      one trial call is allowed and a success closes it.
    * ``select`` can require capability tags and cap estimated cost or latency, and a ``Budget`` keeps total spend
      inside a limit. If nothing is eligible, ``fallback`` is used, else ``NoEligibleArm`` is raised.
    * ``prior`` is another router (typically the global one) whose averages are blended in with weight
      ``prior_weight``, so a new segment starts from what is known instead of from nothing.
    * Pass ``log`` (a list or any ``append``-able sink) to record every outcome as a ``LogRecord`` with the
      probability the policy had of choosing that arm, for later off-policy evaluation.

    Everything is seeded and deterministic. No clocks, no network.
    """

    def __init__(self, arms: Sequence[str | ArmSpec], exploration: float = 1.0, cost_weight: float = 0.0, *,
                 policy: Policy | None = None, discount: float = 1.0, seed: int = 0, trip_after: int = 0, cooldown: int = 20,
                 prior: Router | None = None, prior_weight: float = 5.0, budget: Budget | None = None,
                 fallback: str | None = None, log: Any = None) -> None:
        if not arms:
            raise ValueError("need at least one arm")
        specs = [a if isinstance(a, ArmSpec) else ArmSpec(a) for a in arms]
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("arm names must be unique")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if fallback is not None and fallback not in {s.name for s in specs}:
            raise ValueError("fallback must be one of the arms")
        self.specs = {s.name: s for s in specs}
        self.arms = [s.name for s in specs]
        self.exploration, self.cost_weight, self.discount = exploration, cost_weight, discount
        self.policy: Policy = policy or UCB1()
        self.seed, self.trip_after, self.cooldown = seed, trip_after, cooldown
        self.prior, self.prior_weight, self.budget, self.fallback, self.log = prior, prior_weight, budget, fallback, log
        self.rng = random.Random(seed)
        self.stats = {a: ArmStats() for a in self.arms}
        self.rounds = 0  # outcomes recorded
        self.selections = 0  # select() calls, which is the breaker's clock
        self.last: Decision | None = None

    # ------------------------------------------------------------------------------------------- reading
    @property
    def pulls(self) -> dict[str, int]:
        return {a: self.stats[a].pulls for a in self.arms}

    @property
    def totals(self) -> dict[str, float]:
        return {a: self.stats[a].total for a in self.arms}

    def means(self) -> dict[str, float | None]:
        return {a: (self.stats[a].total / self.stats[a].pulls if self.stats[a].pulls else None) for a in self.arms}

    def best(self) -> str:
        tried = {a: m for a, m in self.means().items() if m is not None}
        if not tried:
            raise ValueError("no arm has been tried yet")
        return max(tried, key=lambda a: tried[a])

    def effective_mean(self, arm: str) -> float:
        """The (discounted, prior-blended) mean reward the policies rank by. Zero with no data at all."""
        st = self.stats[arm]
        if self.prior is not None:
            pm = self.prior.stats[arm]
            if pm.dn > 0:
                return (st.dsum + self.prior_weight * pm.dsum / pm.dn) / (st.dn + self.prior_weight)
        return st.dsum / st.dn if st.dn > 0 else 0.0

    def cost_estimate(self, arm: str) -> float | None:
        st = self.stats[arm]
        return st.cost.mean if st.cost.n else self.specs[arm].cost_hint

    def latency_estimate(self, arm: str) -> float | None:
        st = self.stats[arm]
        return st.latency.mean if st.latency.n else self.specs[arm].latency_hint

    def is_open(self, arm: str) -> bool:
        until = self.stats[arm].open_until
        return until is not None and self.selections < until

    # ------------------------------------------------------------------------------------------- choosing
    def eligible(self, require: Iterable[str] = (), max_cost: float | None = None, max_latency: float | None = None) -> list[str]:
        need = set(require)
        out = []
        for a in self.arms:
            if self.is_open(a) or not need <= self.specs[a].tags:
                continue
            cost, lat = self.cost_estimate(a), self.latency_estimate(a)
            if max_cost is not None and cost is not None and cost > max_cost:
                continue
            if max_latency is not None and lat is not None and lat > max_latency:
                continue
            if self.budget is not None and cost is not None and cost > self.budget.remaining:
                continue
            out.append(a)
        return out

    def select(self, *, require: Iterable[str] = (), max_cost: float | None = None, max_latency: float | None = None) -> str:
        self.selections += 1
        candidates = self.eligible(require, max_cost, max_latency)
        if not candidates:
            if self.fallback is not None:
                self.last = Decision(self.selections, self.fallback, (), None)
                return self.fallback
            raise NoEligibleArm("no arm satisfies the constraints")
        fresh = [a for a in candidates if self.stats[a].pulls == 0 and not (self.prior and self.prior.stats[a].pulls)]
        arm = fresh[0] if fresh else self.policy.choose(self, candidates)
        propensity: float | None = None
        if self.log is not None:
            probs = {a: (1.0 if a == arm else 0.0) for a in candidates} if fresh else self.policy.probabilities(self, candidates)
            propensity = probs.get(arm)
        self.last = Decision(self.selections, arm, tuple(candidates), propensity)
        return arm

    # ------------------------------------------------------------------------------------------- learning
    def update(self, arm: str, quality: float, cost: float = 0.0, *, latency: float | None = None, ok: bool = True,
               context: str | None = None, spend_budget: bool = True) -> float:
        """Record one outcome and return its net reward. A failed call (``ok=False``) counts as quality 0."""
        if arm not in self.stats:
            raise KeyError(arm)
        if cost < 0 or not math.isfinite(cost) or not math.isfinite(quality):
            raise ValueError("quality must be finite and cost must be finite and not negative")
        quality = quality if ok else 0.0
        reward = quality - self.cost_weight * cost
        if self.discount < 1.0:
            for s in self.stats.values():
                s.dn *= self.discount
                s.dsum *= self.discount
        st = self.stats[arm]
        st.pulls += 1
        st.dn += 1.0
        st.dsum += reward
        st.total += reward
        st.reward.add(reward)
        st.quality.add(quality)
        st.cost.add(cost)
        if latency is not None:
            st.latency.add(latency)
        if ok:
            st.consecutive_failures, st.open_until = 0, None
        else:
            st.failures += 1
            st.consecutive_failures += 1
            if self.trip_after and st.consecutive_failures >= self.trip_after:
                st.open_until = self.selections + self.cooldown
        if self.budget is not None and spend_budget:
            self.budget.spent += cost
        self.rounds += 1
        if self.log is not None:
            propensity = self.last.propensity if self.last is not None and self.last.arm == arm else None
            self.log.append(LogRecord(arm, quality, cost, latency, context, propensity, ok))
        return reward

    # ------------------------------------------------------------------------------------------- saving
    def to_dict(self) -> dict[str, Any]:
        version, internal, gauss = self.rng.getstate()
        return {
            "version": 1,
            "arms": [{"name": s.name, "tags": sorted(s.tags), "cost_hint": s.cost_hint, "latency_hint": s.latency_hint} for s in self.specs.values()],
            "config": {"exploration": self.exploration, "cost_weight": self.cost_weight, "discount": self.discount, "seed": self.seed,
                       "trip_after": self.trip_after, "cooldown": self.cooldown, "prior_weight": self.prior_weight, "fallback": self.fallback},
            "policy": self.policy.to_dict(),
            "budget": None if self.budget is None else {"limit": self.budget.limit, "spent": self.budget.spent},
            "rounds": self.rounds, "selections": self.selections, "rng": [version, list(internal), gauss],
            "stats": {a: {"pulls": s.pulls, "dn": s.dn, "dsum": s.dsum, "total": s.total, "reward": s.reward.to_list(),
                          "quality": s.quality.to_list(), "cost": s.cost.to_list(), "latency": s.latency.to_list(),
                          "failures": s.failures, "consecutive_failures": s.consecutive_failures, "open_until": s.open_until}
                      for a, s in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, prior: Router | None = None, log: Any = None) -> Router:
        if data.get("version") != 1:
            raise ValueError("unsupported router state")
        cfg = data["config"]
        arms = [ArmSpec(a["name"], frozenset(a["tags"]), a["cost_hint"], a["latency_hint"]) for a in data["arms"]]
        budget = Budget(**data["budget"]) if data["budget"] else None
        r = cls(arms, cfg["exploration"], cfg["cost_weight"], policy=policy_from_dict(data["policy"]), discount=cfg["discount"],
                seed=cfg["seed"], trip_after=cfg["trip_after"], cooldown=cfg["cooldown"], prior=prior, prior_weight=cfg["prior_weight"],
                budget=budget, fallback=cfg["fallback"], log=log)
        r.rounds, r.selections = data["rounds"], data["selections"]
        version, internal, gauss = data["rng"]
        r.rng.setstate((version, tuple(internal), gauss))
        for a, s in data["stats"].items():
            r.stats[a] = ArmStats(s["pulls"], s["dn"], s["dsum"], s["total"], Welford.from_list(s["reward"]), Welford.from_list(s["quality"]),
                                  Welford.from_list(s["cost"]), Welford.from_list(s["latency"]), s["failures"], s["consecutive_failures"],
                                  s["open_until"])
        return r


class ContextRouter:
    """One router per segment of traffic (task type, tenant, language), each starting from the global picture.

    ``key(context)`` maps whatever you route on to a segment name. A new segment blends the global averages in
    through ``prior_weight`` pseudo-observations, so it does not start cold and does not need a thousand calls
    before it can specialise.
    """

    def __init__(self, arms: Sequence[str | ArmSpec], key: Callable[[Any], str] = str, *, prior_weight: float = 5.0,
                 budget: Budget | None = None, log: Any = None, **kwargs: Any) -> None:
        self._arms, self._kwargs, self.key = list(arms), kwargs, key
        self.prior_weight, self.budget, self.log = prior_weight, budget, log
        self.global_router = Router(self._arms, budget=budget, **kwargs)
        self.segments: dict[str, Router] = {}

    def router_for(self, context: Any) -> Router:
        name = self.key(context)
        if name not in self.segments:
            self.segments[name] = Router(self._arms, prior=self.global_router, prior_weight=self.prior_weight, budget=self.budget,
                                         log=self.log, **self._kwargs)
        return self.segments[name]

    def select(self, context: Any, **constraints: Any) -> str:
        return self.router_for(context).select(**constraints)

    def update(self, context: Any, arm: str, quality: float, cost: float = 0.0, **kw: Any) -> float:
        reward = self.router_for(context).update(arm, quality, cost, context=self.key(context), **kw)
        kw.pop("context", None)
        self.global_router.update(arm, quality, cost, spend_budget=False, **kw)
        return reward
