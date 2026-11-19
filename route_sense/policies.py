"""Selection policies. Each picks one arm from the eligible candidates and can state how likely it was to pick each."""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

from .stats import hoeffding_radius

if TYPE_CHECKING:
    from .router import Router


class Policy:
    name = "policy"

    def choose(self, router: Router, candidates: list[str]) -> str:
        raise NotImplementedError

    def probabilities(self, router: Router, candidates: list[str]) -> dict[str, float]:
        """The probability of choosing each candidate right now, for off-policy evaluation of the logged decisions."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name}


def _best(scores: dict[str, float], order: list[str]) -> str:
    return max(order, key=lambda a: scores[a])  # max keeps the first of equals, so ties break by arm order


class UCB1(Policy):
    """Upper confidence bound on net reward. Deterministic: ties break by arm order. Optimism in the face of doubt."""

    name = "ucb1"

    def choose(self, router: Router, candidates: list[str]) -> str:
        total = max(sum(s.dn for s in router.stats.values()), 1.0)
        scores = {}
        for a in candidates:
            st = router.stats[a]
            scores[a] = router.effective_mean(a) + router.exploration * math.sqrt(math.log(total) / max(st.dn, 1e-12))
        return _best(scores, candidates)

    def probabilities(self, router: Router, candidates: list[str]) -> dict[str, float]:
        pick = self.choose(router, candidates)
        return {a: 1.0 if a == pick else 0.0 for a in candidates}


class Thompson(Policy):
    """Sample each arm's mean from its posterior and take the best draw. Explores in proportion to real doubt."""

    name = "thompson"

    def __init__(self, prior_sd: float = 0.25, draws: int = 400) -> None:
        self.prior_sd, self.draws = prior_sd, draws

    def _sample(self, router: Router, a: str, rng: random.Random) -> float:
        st = router.stats[a]
        sd = max(st.reward.sd if st.reward.n > 1 else self.prior_sd, 0.02)
        return rng.gauss(router.effective_mean(a), sd / math.sqrt(max(st.dn, 1.0)))

    def choose(self, router: Router, candidates: list[str]) -> str:
        draws = {a: self._sample(router, a, router.rng) for a in candidates}
        return _best(draws, candidates)

    def probabilities(self, router: Router, candidates: list[str]) -> dict[str, float]:
        rng = random.Random(router.selections * 7919 + 1)  # reproducible, and independent of the decision stream
        wins = dict.fromkeys(candidates, 0)
        for _ in range(self.draws):
            wins[_best({a: self._sample(router, a, rng) for a in candidates}, candidates)] += 1
        return {a: wins[a] / self.draws for a in candidates}

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "prior_sd": self.prior_sd, "draws": self.draws}


class EpsilonGreedy(Policy):
    """Best mean most of the time, a uniformly random candidate with probability ``epsilon``. Exact propensities."""

    name = "epsilon"

    def __init__(self, epsilon: float = 0.1) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon

    def _greedy(self, router: Router, candidates: list[str]) -> str:
        return _best({a: router.effective_mean(a) for a in candidates}, candidates)

    def choose(self, router: Router, candidates: list[str]) -> str:
        if router.rng.random() < self.epsilon:
            return router.rng.choice(candidates)
        return self._greedy(router, candidates)

    def probabilities(self, router: Router, candidates: list[str]) -> dict[str, float]:
        greedy, k = self._greedy(router, candidates), len(candidates)
        return {a: self.epsilon / k + ((1 - self.epsilon) if a == greedy else 0.0) for a in candidates}

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "epsilon": self.epsilon}


class CheapestAboveFloor(Policy):
    """The cheapest arm that is not yet proven to fall below a quality floor.

    The question a finance team actually asks: "cheapest model that is good enough". Quality is judged on its own
    (not net of cost) with a Hoeffding interval, so an arm is *proven* safe when its lower bound clears the floor
    and *disproven* when its upper bound does not. The policy picks the cheapest arm that is not disproven, which
    gets cheap arms enough traffic to be proven or ruled out. With ``strict`` it only ever picks proven arms and
    otherwise falls back to the highest-quality arm.
    """

    name = "cheapest_above_floor"

    def __init__(self, floor: float, delta: float = 0.05, strict: bool = False) -> None:
        if not 0 <= floor <= 1:
            raise ValueError("floor must be between 0 and 1")
        self.floor, self.delta, self.strict = floor, delta, strict

    def bounds(self, router: Router, a: str) -> tuple[float, float]:
        q = router.stats[a].quality
        r = hoeffding_radius(q.n, self.delta)
        return q.mean - r, q.mean + r

    def proven(self, router: Router, a: str) -> bool:
        return self.bounds(router, a)[0] >= self.floor

    def choose(self, router: Router, candidates: list[str]) -> str:
        def cost_key(a: str) -> float:
            c = router.cost_estimate(a)
            return math.inf if c is None else c

        by_cost = sorted(candidates, key=cost_key)
        pool = [a for a in by_cost if self.proven(router, a)] if self.strict else [a for a in by_cost if self.bounds(router, a)[1] >= self.floor]
        if pool:
            return pool[0]
        return max(candidates, key=lambda a: router.stats[a].quality.mean)

    def probabilities(self, router: Router, candidates: list[str]) -> dict[str, float]:
        pick = self.choose(router, candidates)
        return {a: 1.0 if a == pick else 0.0 for a in candidates}

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "floor": self.floor, "delta": self.delta, "strict": self.strict}


def policy_from_dict(data: dict[str, Any]) -> Policy:
    kind = data["type"]
    if kind == "ucb1":
        return UCB1()
    if kind == "thompson":
        return Thompson(data.get("prior_sd", 0.25), data.get("draws", 400))
    if kind == "epsilon":
        return EpsilonGreedy(data.get("epsilon", 0.1))
    if kind == "cheapest_above_floor":
        return CheapestAboveFloor(data["floor"], data.get("delta", 0.05), data.get("strict", False))
    raise ValueError(f"unknown policy {kind!r}")
