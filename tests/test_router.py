import unittest

from route_sense import ArmSpec, Budget, ContextRouter, NoEligibleArm, Router, Thompson


class ConstraintTests(unittest.TestCase):
    def test_tags_filter_arms(self):
        r = Router([ArmSpec("text", frozenset({"text"})), ArmSpec("vision", frozenset({"text", "vision"}))])
        self.assertEqual(r.eligible(require=["vision"]), ["vision"])
        self.assertEqual(r.select(require=["vision"]), "vision")

    def test_max_cost_uses_hints_then_observations(self):
        r = Router([ArmSpec("cheap", cost_hint=0.1), ArmSpec("pricey", cost_hint=2.0)])
        self.assertEqual(r.eligible(max_cost=0.5), ["cheap"])
        r.update("pricey", 0.9, 0.2)  # a real observation beats the hint
        self.assertEqual(r.eligible(max_cost=0.5), ["cheap", "pricey"])

    def test_max_latency(self):
        r = Router([ArmSpec("fast", latency_hint=0.2), ArmSpec("slow", latency_hint=5.0)])
        self.assertEqual(r.eligible(max_latency=1.0), ["fast"])

    def test_budget_excludes_arms_that_no_longer_fit_and_is_spent_by_updates(self):
        budget = Budget(1.0)
        r = Router([ArmSpec("cheap", cost_hint=0.1), ArmSpec("pricey", cost_hint=0.8)], budget=budget)
        r.update("pricey", 0.9, 0.8)
        self.assertAlmostEqual(budget.remaining, 0.2)
        self.assertEqual(r.eligible(), ["cheap"])
        r.update("cheap", 0.5, 0.2)
        self.assertTrue(budget.exhausted)

    def test_nothing_eligible_raises_or_uses_the_fallback(self):
        arms = [ArmSpec("a", cost_hint=5.0), ArmSpec("b", cost_hint=6.0)]
        with self.assertRaises(NoEligibleArm):
            Router(arms).select(max_cost=1.0)
        self.assertEqual(Router(arms, fallback="b").select(max_cost=1.0), "b")
        with self.assertRaises(ValueError):
            Router(arms, fallback="zzz")

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            Router(["a", "a"])
        with self.assertRaises(ValueError):
            Router(["a"], discount=0)
        r = Router(["a"])
        with self.assertRaises(ValueError):
            r.update("a", 0.5, -1)
        with self.assertRaises(ValueError):
            r.update("a", float("nan"))
        with self.assertRaises(KeyError):
            r.update("zzz", 0.5)
        with self.assertRaises(ValueError):
            r.best()


class BreakerTests(unittest.TestCase):
    def test_trips_cools_down_half_opens_and_closes(self):
        r = Router(["a", "b"], trip_after=3, cooldown=5)
        r.update("a", 0.9)
        r.update("b", 0.5)
        for _ in range(3):
            r.select()
            r.update("a", 0.0, ok=False)
        self.assertTrue(r.is_open("a"))
        for _ in range(4):
            self.assertEqual(r.select(), "b")  # a is out while the breaker is open
            r.update("b", 0.5)
        r.select()  # cooldown over: a is eligible again
        self.assertFalse(r.is_open("a"))
        r.update("a", 0.9)  # one success closes it for good
        self.assertEqual((r.stats["a"].consecutive_failures, r.stats["a"].open_until), (0, None))

    def test_all_open_cooldown_still_advances(self):
        r = Router(["a"], trip_after=1, cooldown=3)
        r.update("a", 0, ok=False)
        raised = 0
        for _ in range(5):
            try:
                r.select()
            except NoEligibleArm:
                raised += 1
        self.assertEqual(raised, 2)  # selections 1 and 2 are inside the cooldown, 3 onwards are not

    def test_a_failed_call_counts_as_zero_quality(self):
        r = Router(["a"], cost_weight=1.0)
        self.assertAlmostEqual(r.update("a", 0.9, 0.25, ok=False), -0.25)


class PriorTests(unittest.TestCase):
    def test_a_segment_starts_from_the_global_picture(self):
        glob = Router(["a", "b"])
        for _ in range(50):
            glob.update("a", 0.3)
            glob.update("b", 0.8)
        seg = Router(["a", "b"], prior=glob, prior_weight=5)
        self.assertEqual(seg.select(), "b")  # no forced exploration: the prior already has data
        for _ in range(3):
            seg.update("a", 1.0)
        self.assertAlmostEqual(seg.effective_mean("a"), (3.0 + 5 * 0.3) / 8)

    def test_context_router_segments_learn_separately_and_share_a_budget(self):
        budget = Budget(10.0)
        cr = ContextRouter(["a", "b"], key=lambda ctx: ctx["task"], budget=budget)
        for _ in range(40):
            for task, best in (("code", "a"), ("chat", "b")):
                arm = cr.select({"task": task})
                cr.update({"task": task}, arm, 0.9 if arm == best else 0.2, 0.01)
        self.assertEqual(cr.router_for({"task": "code"}).best(), "a")
        self.assertEqual(cr.router_for({"task": "chat"}).best(), "b")
        self.assertAlmostEqual(budget.spent, 0.8)  # 80 calls, each charged once, not once per router


class StateTests(unittest.TestCase):
    def test_round_trip_continues_identically(self):
        a = Router([ArmSpec("x", frozenset({"t"}), 0.1, 0.5), "y"], policy=Thompson(), discount=0.97, seed=9, budget=Budget(5.0), trip_after=2)
        for i in range(30):
            arm = a.select()
            a.update(arm, (i % 5) / 5, 0.01, latency=0.3, ok=i % 7 != 0)
        b = Router.from_dict(a.to_dict())
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual([a.select() for _ in range(20)], [b.select() for _ in range(20)])  # same random stream too

    def test_rejects_unknown_state(self):
        with self.assertRaises(ValueError):
            Router.from_dict({"version": 99})


class LogTests(unittest.TestCase):
    def test_log_records_outcomes_with_propensity(self):
        log: list = []
        r = Router(["a", "b"], log=log)
        for _ in range(4):
            arm = r.select()
            r.update(arm, 0.5, 0.1, latency=0.2)
        self.assertEqual(len(log), 4)
        self.assertTrue(all(rec.propensity == 1.0 for rec in log))  # UCB is deterministic

    def test_default_has_no_overhead_or_log(self):
        r = Router(["a"])
        r.update(r.select(), 1.0)
        self.assertIsNone(r.last.propensity)


if __name__ == "__main__":
    unittest.main()
