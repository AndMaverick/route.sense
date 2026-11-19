"""route.sense: model routing that learns which model earns its cost, within the limits you set.

Bandit policies (UCB1, Thompson, epsilon-greedy, cheapest-above-a-quality-floor), drift forgetting, circuit breakers,
tag, cost, latency and budget constraints, segment routing with shrinkage, and offline evaluation from logs.
"""
from __future__ import annotations

from .offline import (
    ArmReport,
    Comparison,
    IpsResult,
    ReplayResult,
    arm_report,
    compare,
    ips,
    replay,
)
from .policies import UCB1, CheapestAboveFloor, EpsilonGreedy, Policy, Thompson
from .router import (
    ArmSpec,
    Budget,
    ContextRouter,
    Decision,
    LogRecord,
    NoEligibleArm,
    Router,
)
from .sim import Environment, SimResult, run

__version__ = "0.2.0"

__all__ = [
    "UCB1",
    "ArmReport",
    "ArmSpec",
    "Budget",
    "CheapestAboveFloor",
    "Comparison",
    "ContextRouter",
    "Decision",
    "Environment",
    "EpsilonGreedy",
    "IpsResult",
    "LogRecord",
    "NoEligibleArm",
    "Policy",
    "ReplayResult",
    "Router",
    "SimResult",
    "Thompson",
    "__version__",
    "arm_report",
    "compare",
    "ips",
    "replay",
    "run",
]
