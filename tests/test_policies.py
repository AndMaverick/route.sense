import unittest

from route_sense import UCB1, CheapestAboveFloor, Environment, EpsilonGreedy, Router, Thompson, run

ENV = Environment({"small": 0.60, "large": 0.90, "huge": 0.91}, {"small": 0.1, "large": 0.4, "huge": 3.0})


class PolicyTests(unittest.TestCase):
    def test_every_policy_finds_the_best_net_arm_with_sublinear_regret(self):
        # at cost_weight 0.5 the net rewards are small .55, large .70, huge -.59: large is best
        for policy in (UCB1(), Thompson(), EpsilonGreedy(0.05)):
            router = Router(["small", "large", "huge"], cost_weight=0.5, policy=policy, seed=3)
            res = run(router, ENV, 3000, seed=11)
            self.assertEqual(max(res.pulls, key=lambda a: res.pulls[a]), "large", policy.name)
            if policy.name != "epsilon":  # constant-epsilon exploration has linear regret by design
                second_half = res.regret_curve[-1] - res.regret_curve[1499]
                self.assertLess(second_half, res.regret_curve[1499] * 0.8, policy.name)  # regret growth slows

    def test_same_seed_same_run(self):
        a = run(Router(["small", "large"], policy=Thompson(), seed=5), Environment({"small": .5, "large": .7}, {"small": 0, "large": 0}), 300, seed=2)
        b = run(Router(["small", "large"], policy=Thompson(), seed=5), Environment({"small": .5, "large": .7}, {"small": 0, "large": 0}), 300, seed=2)
        self.assertEqual((a.pulls, a.regret), (b.pulls, b.regret))

    def test_epsilon_probabilities_are_exact_and_sum_to_one(self):
        r = Router(["a", "b", "c"], policy=EpsilonGreedy(0.3))
        for arm, q in (("a", 0.2), ("b", 0.9), ("c", 0.5)):
            r.select()
            r.update(arm, q)
        probs = r.policy.probabilities(r, ["a", "b", "c"])
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs["b"], 0.1 + 0.7)
        self.assertAlmostEqual(probs["a"], 0.1)
        with self.assertRaises(ValueError):
            EpsilonGreedy(1.5)

    def test_thompson_probabilities_favour_the_better_arm(self):
        r = Router(["a", "b"], policy=Thompson())
        for _ in range(30):
            r.update("a", 0.3)
            r.update("b", 0.8)
        p = r.policy.probabilities(r, ["a", "b"])
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertGreater(p["b"], 0.95)

    def test_drift_forgetting_notices_a_model_that_gets_worse(self):
        # "large" is better until round 600, then a silent regression makes "small" better
        def drift(t):
            return {"small": 0.6, "large": 0.9} if t < 600 else {"small": 0.6, "large": 0.2}
        env = Environment({"small": .6, "large": .9}, {"small": 0, "large": 0}, drift)
        forgetful = run(Router(["small", "large"], discount=0.98, seed=1), env, 1500, seed=4)
        stubborn = run(Router(["small", "large"], seed=1), env, 1500, seed=4)
        self.assertLess(forgetful.regret, stubborn.regret)


class FloorPolicyTests(unittest.TestCase):
    def build(self, floor, strict=False):
        return Router(["cheap", "mid", "pricey"], cost_weight=0.0, policy=CheapestAboveFloor(floor, strict=strict))

    def feed(self, r, n=400):
        env = Environment({"cheap": 0.70, "mid": 0.88, "pricey": 0.95}, {"cheap": 0.1, "mid": 0.5, "pricey": 2.0})
        return run(r, env, n, seed=9)

    def test_picks_the_cheapest_arm_that_clears_the_floor(self):
        res = self.feed(self.build(0.85))
        self.assertEqual(max(res.pulls, key=lambda a: res.pulls[a]), "mid")
        self.assertLess(res.pulls["pricey"], 40)  # explored, then abandoned

    def test_a_lower_floor_moves_traffic_to_the_cheap_arm(self):
        res = self.feed(self.build(0.6))
        self.assertEqual(max(res.pulls, key=lambda a: res.pulls[a]), "cheap")

    def test_strict_mode_only_routes_to_proven_arms_after_the_first_pass(self):
        r = self.build(0.85, strict=True)
        self.feed(r, 600)
        pol = r.policy
        self.assertTrue(pol.proven(r, "pricey") or pol.proven(r, "mid"))
        self.assertFalse(pol.proven(r, "cheap"))

    def test_unreachable_floor_falls_back_to_best_quality(self):
        res = self.feed(self.build(0.99))
        self.assertEqual(max(res.pulls, key=lambda a: res.pulls[a]), "pricey")

    def test_floor_validated(self):
        with self.assertRaises(ValueError):
            CheapestAboveFloor(1.5)


if __name__ == "__main__":
    unittest.main()
