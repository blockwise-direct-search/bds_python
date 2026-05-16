import os
import unittest

import numpy as np

from bds import minimize_bds


ALGORITHMS = ("cbds", "pbds", "rbds", "pads", "ds")


def _selected_algorithms():
    algorithm = os.environ.get("BDS_TEST_ALGORITHM")
    return (algorithm,) if algorithm else ALGORITHMS


def _recursive_objective(x, depth):
    value = float((x[0] - 1.0) ** 2)
    if depth <= 0:
        return value

    inner = minimize_bds(
        lambda y: _recursive_objective(y, depth - 1),
        np.array([0.0]),
        options={
            "Algorithm": "ds",
            "MaxFunctionEvaluations": 40,
            "StepTolerance": 1e-4,
            "seed": depth,
        },
    )
    return value + 0.01 * inner.fun


class RecursiveInvocationTests(unittest.TestCase):
    def test_solver_can_be_called_recursively_from_objective(self):
        depth = int(os.environ.get("BDS_TEST_DEPTH", "1"))

        for algorithm in _selected_algorithms():
            with self.subTest(algorithm=algorithm):
                result = minimize_bds(
                    lambda x: _recursive_objective(x, depth),
                    np.array([0.0]),
                    options={
                        "Algorithm": algorithm,
                        "MaxFunctionEvaluations": 300,
                        "StepTolerance": 1e-6,
                        "seed": 0,
                    },
                )
                self.assertLess(result.fun, 1e-8)
                np.testing.assert_allclose(result.x, [1.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
