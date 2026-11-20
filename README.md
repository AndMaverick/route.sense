# route.sense

[![CI](https://github.com/AndMaverick/route.sense/actions/workflows/ci.yml/badge.svg)](https://github.com/AndMaverick/route.sense/actions/workflows/ci.yml)

Model routing that learns. Each model is an arm of a bandit, and reward is quality minus a weighted cost, so the cheap model wins when it is good enough and the expensive one wins when the quality gap pays for itself. Around that core: the question finance actually asks ("what is the cheapest model that clears our quality bar?"), forgetting for models that change under you, circuit breakers, budgets, segment routing, and offline evaluation from logs you already have. Standard library only, deterministic, typed, Python 3.10 and up.

```
pip install git+https://github.com/AndMaverick/route.sense
```

## Use

```python
from route_sense import Router

router = Router(["small", "large"], cost_weight=0.5)

for request in traffic:
    model = router.select()
    answer, quality, cost = call(model, request)   # quality from your eval or feedback signal
    router.update(model, quality, cost)

router.best()    # the model currently earning the most per call
router.means()   # average net reward per model
```

## The cheapest model that is good enough

```python
from route_sense import Router, CheapestAboveFloor

router = Router(["small", "mid", "large"], policy=CheapestAboveFloor(floor=0.85))
```

Quality is judged on its own, with a Hoeffding interval. An arm is **proven** when its lower bound clears the floor and **disproven** when its upper bound cannot. The router sends traffic to the cheapest arm that is not disproven, so cheap models get enough calls to be proven or ruled out, and traffic settles on the cheapest one that holds the line. `strict=True` only ever routes to proven arms. Raise or lower the floor and the traffic moves, which is the number a CFO can set.

## Policies

| Policy | Behaviour |
|---|---|
| `UCB1()` (default) | Deterministic. Optimism under uncertainty, ties break by arm order. |
| `Thompson()` | Samples each arm's mean from its posterior. Explores in proportion to real doubt. |
| `EpsilonGreedy(0.1)` | Exact, known probabilities, which makes logs easy to evaluate offline. |
| `CheapestAboveFloor(f)` | Cheapest arm not yet proven to fall below quality `f`. |

## Running it for real

```python
from route_sense import Router, ArmSpec, Budget

router = Router(
    [ArmSpec("small", frozenset({"text"}), cost_hint=0.001),
     ArmSpec("large", frozenset({"text", "vision"}), cost_hint=0.02)],
    discount=0.99,                 # forget slowly: a model that quietly gets worse is noticed
    trip_after=3, cooldown=50,     # three failures in a row open a breaker for 50 selections
    budget=Budget(limit=250.0),    # keep total spend inside a limit
    fallback="large",              # used when every arm is excluded
)

model = router.select(require=["vision"], max_cost=0.05, max_latency=4.0)
```

- **Forgetting.** `discount` below 1 gives less weight to old observations, so a provider changing a model is a tracking problem, not a permanent mistake. The test suite simulates a silent regression and compares regret with and without it.
- **Circuit breaker.** A failed call counts as quality 0. After `trip_after` in a row the arm is out for `cooldown` selections, then gets one trial call. A success closes it.
- **Constraints.** Tags, cost and latency caps use your hints until real observations replace them. A `Budget` keeps spend inside a limit. If nothing is eligible you get `fallback`, or `NoEligibleArm`.
- **Segments.** `ContextRouter` keeps one router per task type or tenant, each starting from the global averages through a prior, so a new segment does not start cold.
- **State.** `router.to_dict()` and `Router.from_dict()` round trip everything, including the random stream, so a restored router continues exactly where it stopped.

## Learn from the logs you already have

Pass `log=[]` to a router and every outcome is recorded with the probability the policy had of choosing that model. Or bring your own JSONL (`arm`, `quality`, `cost`).

```
python -m route_sense report calls.jsonl --baseline large --cost-weight 0.2
python -m route_sense replay calls.jsonl --policy floor --floor 0.85
```

`report` gives quality, cost and net reward per model, cost and net reward with intervals, and a significance test against the baseline, including cost per 1000 calls if traffic moved. Two estimators answer "what would this policy have done on past traffic", and their assumptions are the point:

- `replay` is unbiased only if the logged model for each call was chosen uniformly at random. A short randomised rollout gives you that log.
- `ips` works on any log that recorded `propensity`, including one from an adaptive router. It is normalised by itself and clips large weights, which trades a little bias for much less variance.

Neither says anything about a model that never appears in the log.

## Testing a router before it sees traffic

```python
from route_sense import Environment, run

env = Environment(quality={"small": .6, "large": .9}, cost={"small": .1, "large": .4})
result = run(Router(["small", "large"], cost_weight=0.5), env, rounds=3000)
result.regret, result.pulls
```

## Honest limits

- It is only as good as the quality signal you feed it. A noisy or biased score teaches the router the wrong lesson efficiently.
- Exploration costs something: until an arm is proven or disproven, some calls go to it. `strict` mode limits this at the price of learning more slowly.
- The significance tests are normal approximations and treat calls as independent. Do not read a p value as more than a sanity check on a few thousand calls.
- Prices are an input you supply. Nothing here knows what any provider charges.

Pairs with [eval.forge](https://github.com/AndMaverick/eval.forge) for the quality signal and [model.ledger](https://github.com/AndMaverick/model.ledger) for the cost signal.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Read the thinking:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=route.sense)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=route.sense).
