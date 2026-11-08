# route.sense

Model routing that learns. Instead of hardcoding "use the big model for hard tasks", route.sense treats each model as an arm of a bandit and picks by upper confidence bound on net reward, where reward is quality minus a weighted cost. The cheap model wins when it is good enough, and the expensive one wins when the quality gap pays for itself. Standard library only, fully deterministic.

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

## Design

- **UCB1.** Every arm is tried once, then the router balances proven reward against uncertainty, so a model with little data still gets a fair chance.
- **Cost aware.** `cost_weight` converts spend into the same units as quality. At 0 the router chases quality. Raise it and cheaper models win more of the traffic. The tests show the winner flipping on this one number.
- **Deterministic.** No randomness. Ties break by arm order, so runs are reproducible and easy to debug.
- **Pairs with** [eval.forge](https://github.com/AndMaverick/eval.forge) for the quality signal and [model.ledger](https://github.com/AndMaverick/model.ledger) for the cost signal.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related reading:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=route.sense)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=route.sense).
