"""route.sense: a UCB1 router that learns which model earns its cost."""

from __future__ import annotations

import math

__version__ = "0.1.0"


class Router:
    """Pick among arms (models) by upper confidence bound on net reward.

    Reward for one call is quality minus cost_weight times cost, so a model that
    is slightly better but much pricier loses. Deterministic: ties break by arm order.
    """

    def __init__(self, arms, exploration: float = 1.0, cost_weight: float = 0.0) -> None:
        if not arms:
            raise ValueError("need at least one arm")
        self.arms = list(arms)
        self.exploration = exploration
        self.cost_weight = cost_weight
        self.pulls = {a: 0 for a in self.arms}
        self.totals = {a: 0.0 for a in self.arms}
        self.rounds = 0

    def select(self) -> str:
        for arm in self.arms:
            if self.pulls[arm] == 0:
                return arm
        def ucb(arm):
            mean = self.totals[arm] / self.pulls[arm]
            bonus = self.exploration * math.sqrt(math.log(self.rounds) / self.pulls[arm])
            return mean + bonus
        return max(self.arms, key=ucb)

    def update(self, arm: str, quality: float, cost: float = 0.0) -> float:
        reward = quality - self.cost_weight * cost
        self.pulls[arm] += 1
        self.totals[arm] += reward
        self.rounds += 1
        return reward

    def means(self) -> dict:
        return {a: (self.totals[a] / self.pulls[a] if self.pulls[a] else None) for a in self.arms}

    def best(self) -> str:
        tried = {a: m for a, m in self.means().items() if m is not None}
        return max(tried, key=tried.get)
