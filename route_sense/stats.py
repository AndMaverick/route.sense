"""Small statistics used by the router and the offline tools: online mean and variance, intervals, a z test."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Welford:
    """Running mean and variance in one pass, without storing the observations."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    @property
    def sem(self) -> float:
        return self.sd / math.sqrt(self.n) if self.n > 1 else math.inf

    def to_list(self) -> list[float]:
        return [self.n, self.mean, self.m2]

    @classmethod
    def from_list(cls, data: list[float]) -> Welford:
        return cls(int(data[0]), float(data[1]), float(data[2]))


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def hoeffding_radius(n: int, delta: float = 0.05) -> float:
    """Half-width of a (1 - delta) confidence interval for the mean of n draws bounded in [0, 1]. Infinite for n = 0."""
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    return math.inf if n <= 0 else math.sqrt(math.log(1.0 / delta) / (2.0 * n))
