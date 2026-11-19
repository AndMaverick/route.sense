"""A small simulator for testing a router before it sees traffic: arms with known quality, cost and optional drift."""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .router import Router


@dataclass
class Environment:
    """Each arm returns quality 1 with probability ``quality[arm]`` (else 0) at ``cost[arm]``. ``drift(round)``, if
    given, returns a replacement quality table for that round, which models a provider quietly changing a model."""

    quality: dict[str, float]
    cost: dict[str, float]
    drift: Callable[[int], dict[str, float] | None] | None = None

    def table(self, round_no: int) -> dict[str, float]:
        return (self.drift(round_no) if self.drift else None) or self.quality


@dataclass
class SimResult:
    regret: float
    pulls: dict[str, int]
    mean_reward: float
    regret_curve: list[float] = field(repr=False, default_factory=list)


def run(router: Router, env: Environment, rounds: int, seed: int = 1) -> SimResult:
    """Drive ``router`` for ``rounds`` calls. Regret is measured against the arm with the best expected net reward."""
    rng = random.Random(seed)
    regret, total, curve = 0.0, 0.0, []
    for t in range(rounds):
        table = env.table(t)
        expected = {a: table[a] - router.cost_weight * env.cost[a] for a in router.arms}
        arm = router.select()
        quality = 1.0 if rng.random() < table[arm] else 0.0
        total += router.update(arm, quality, env.cost[arm])
        regret += max(expected.values()) - expected[arm]
        curve.append(regret)
    return SimResult(regret, router.pulls, total / max(rounds, 1), curve)
