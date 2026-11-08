import unittest

from route_sense import Router

QUALITY = {"small": 0.60, "large": 0.90}
COST = {"small": 0.1, "large": 1.0}


def simulate(router, rounds=400):
    for _ in range(rounds):
        arm = router.select()
        router.update(arm, QUALITY[arm], COST[arm])
    return router


class RouterTests(unittest.TestCase):
    def test_tries_every_arm_first_in_order(self):
        router = Router(["a", "b", "c"])
        seen = []
        for _ in range(3):
            arm = router.select()
            seen.append(arm)
            router.update(arm, 0.5)
        self.assertEqual(seen, ["a", "b", "c"])

    def test_converges_on_higher_quality_when_cost_is_ignored(self):
        router = simulate(Router(["small", "large"], cost_weight=0.0))
        self.assertEqual(router.best(), "large")
        self.assertGreater(router.pulls["large"], router.pulls["small"] * 3)

    def test_cost_weight_flips_the_winner(self):
        router = simulate(Router(["small", "large"], cost_weight=0.5))
        self.assertEqual(router.best(), "small")
        self.assertGreater(router.pulls["small"], router.pulls["large"] * 3)

    def test_update_returns_net_reward(self):
        router = Router(["a"], cost_weight=2.0)
        self.assertAlmostEqual(router.update("a", 0.9, 0.2), 0.5)

    def test_needs_arms(self):
        with self.assertRaises(ValueError):
            Router([])


if __name__ == "__main__":
    unittest.main()
