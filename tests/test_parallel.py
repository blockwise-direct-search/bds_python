import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bds import minimize_bds


ALGORITHMS = ("cbds", "pbds", "rbds", "pads", "ds")


def _selected_algorithms():
    algorithm = os.environ.get("BDS_TEST_ALGORITHM")
    return (algorithm,) if algorithm else ALGORITHMS


def _solve_quadratic(algorithm, seed):
    target = np.array([1.0, -1.0])

    def quadratic(x):
        return float(np.sum((x - target) ** 2))

    return minimize_bds(
        quadratic,
        np.zeros(2),
        options={
            "Algorithm": algorithm,
            "MaxFunctionEvaluations": 800,
            "StepTolerance": 1e-7,
            "seed": seed,
        },
    )


class ParallelInvocationTests(unittest.TestCase):
    def test_multiple_independent_solver_calls_can_run_concurrently(self):
        algorithms = _selected_algorithms()
        seed_text = os.environ.get("BDS_TEST_SEED")
        seeds = (int(seed_text), int(seed_text) + 1) if seed_text else (0, 1)
        tasks = [(algorithm, seed) for algorithm in algorithms for seed in seeds]

        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
            results = list(executor.map(lambda item: _solve_quadratic(*item), tasks))

        for result in results:
            self.assertLess(result.fun, 1e-10)
            np.testing.assert_allclose(result.x, [1.0, -1.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
