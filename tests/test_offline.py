import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout

from route_sense import EpsilonGreedy, LogRecord, Router, arm_report, compare, ips, replay
from route_sense.cli import main
from route_sense.offline import from_jsonl, to_jsonl
from route_sense.stats import Welford, hoeffding_radius, two_sided_p

TRUE_Q = {"small": 0.60, "large": 0.90}
TRUE_C = {"small": 0.1, "large": 0.5}


def uniform_log(n, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        arm = rng.choice(["small", "large"])
        out.append(LogRecord(arm, 1.0 if rng.random() < TRUE_Q[arm] else 0.0, TRUE_C[arm], propensity=0.5))
    return out


class StatsTests(unittest.TestCase):
    def test_welford_matches_the_textbook(self):
        w = Welford()
        for x in (2, 4, 4, 4, 5, 5, 7, 9):
            w.add(x)
        self.assertAlmostEqual(w.mean, 5.0)
        self.assertAlmostEqual(w.var, 32 / 7)
        self.assertEqual(Welford.from_list(w.to_list()), w)

    def test_hoeffding_and_p_value(self):
        self.assertAlmostEqual(hoeffding_radius(100, 0.05), 0.12238, places=4)
        self.assertEqual(hoeffding_radius(0), float("inf"))
        self.assertAlmostEqual(two_sided_p(1.96), 0.05, places=3)
        with self.assertRaises(ValueError):
            hoeffding_radius(5, 1.5)


class ReportTests(unittest.TestCase):
    def test_arm_report(self):
        rep = arm_report(uniform_log(4000), cost_weight=1.0)
        self.assertAlmostEqual(rep["large"].quality, 0.9, delta=0.04)
        self.assertAlmostEqual(rep["large"].reward, 0.4, delta=0.04)
        self.assertLess(rep["large"].reward_low, rep["large"].reward)
        self.assertLess(rep["large"].reward, rep["large"].reward_high)
        self.assertEqual(sum(r.n for r in rep.values()), 4000)

    def test_compare_detects_a_real_difference_and_not_a_fake_one(self):
        log = uniform_log(2000)
        real = compare(log, "small", "large")
        self.assertTrue(real.significant)
        self.assertAlmostEqual(real.quality_diff, 0.3, delta=0.05)
        self.assertAlmostEqual(real.per_1000_cost, 400, delta=5)
        rng = random.Random(4)
        flat = [LogRecord(a, 1.0 if rng.random() < 0.7 else 0.0, 0.1) for a in ["x", "y"] * 500]  # identical arms
        self.assertFalse(compare(flat, "x", "y").significant)
        tiny = [LogRecord("x", 1.0), LogRecord("y", 0.0)]
        self.assertFalse(compare(tiny, "x", "y").significant)  # one call each proves nothing
        with self.assertRaises(ValueError):
            compare(tiny, "x", "nope")

    def test_net_reward_can_reverse_the_winner(self):
        log = uniform_log(3000)
        self.assertGreater(compare(log, "small", "large", cost_weight=0.0).reward_diff, 0)
        self.assertLess(compare(log, "small", "large", cost_weight=1.0).reward_diff, 0.0 + 0.001)

    def test_jsonl_round_trip(self):
        log = uniform_log(5)
        self.assertEqual(from_jsonl(to_jsonl(log)), log)


class EstimatorTests(unittest.TestCase):
    def test_replay_is_close_to_the_truth_on_a_uniform_log(self):
        log = uniform_log(20000)
        res = replay(log, lambda: Router(["small", "large"], cost_weight=0.0))
        # the router settles on "large" (quality .9); early exploration drags the estimate slightly down
        self.assertGreater(res.mean_quality, 0.85)
        self.assertLessEqual(res.mean_quality, 0.9 + 0.02)
        self.assertGreater(res.matched, 5000)

    def test_ips_recovers_the_value_of_a_fixed_policy(self):
        log = uniform_log(20000, seed=3)
        always_large = ips(log, lambda rec: {"large": 1.0}, cost_weight=1.0)
        self.assertAlmostEqual(always_large.value, 0.9 - 0.5, delta=0.02)
        always_small = ips(log, lambda rec: {"small": 1.0}, cost_weight=1.0)
        self.assertAlmostEqual(always_small.value, 0.6 - 0.1, delta=0.02)
        self.assertGreater(always_large.effective_sample_size, 5000)

    def test_ips_works_on_a_log_from_an_adaptive_router(self):
        log: list = []
        router = Router(["small", "large"], policy=EpsilonGreedy(0.3), cost_weight=1.0, seed=2, log=log)
        rng = random.Random(8)
        for _ in range(15000):
            arm = router.select()
            router.update(arm, 1.0 if rng.random() < TRUE_Q[arm] else 0.0, TRUE_C[arm])
        # the log is adaptive, yet inverse weighting still recovers what "always small" would have earned
        est = ips(log, lambda rec: {"small": 1.0}, cost_weight=1.0)
        self.assertAlmostEqual(est.value, 0.5, delta=0.05)

    def test_ips_needs_propensities(self):
        with self.assertRaises(ValueError):
            ips([LogRecord("a", 1.0)], lambda rec: {"a": 1.0})


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "log.jsonl")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(to_jsonl(uniform_log(3000)))

    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_report_with_baseline(self):
        code, out = self.run_cli("report", self.path, "--baseline", "small", "--cost-weight", "0.2")
        self.assertEqual(code, 0)
        self.assertIn("large vs small", out)
        self.assertIn("significant", out)

    def test_report_json(self):
        code, out = self.run_cli("report", self.path, "--json", "--baseline", "small")
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(set(data["arms"]), {"small", "large"})
        self.assertEqual(len(data["comparisons"]), 1)

    def test_replay_and_empty_log_and_bad_baseline(self):
        self.assertIn("matched", self.run_cli("replay", self.path, "--policy", "thompson")[1])
        self.assertIn("matched", self.run_cli("replay", self.path, "--policy", "floor", "--floor", "0.8")[1])
        empty = os.path.join(self.dir.name, "empty.jsonl")
        open(empty, "w").close()
        self.assertEqual(self.run_cli("report", empty)[0], 2)
        self.assertEqual(self.run_cli("report", self.path, "--baseline", "nope")[0], 2)


if __name__ == "__main__":
    unittest.main()
