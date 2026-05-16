import os
import unittest

import numpy as np

from bds import minimize_bds


ALGORITHMS = ("cbds", "pbds", "rbds", "pads", "ds")


def _selected_algorithm():
    return os.environ.get("BDS_TEST_ALGORITHM")


def _is_tough():
    return os.environ.get("BDS_TEST_TOUGH", "false").lower() == "true"


def _dimension():
    if "BDS_TEST_DIMENSION" in os.environ:
        return int(os.environ["BDS_TEST_DIMENSION"])
    return 12 if _is_tough() else 6


def _maxfev(n):
    if "BDS_TEST_MAXFEV" in os.environ:
        return int(os.environ["BDS_TEST_MAXFEV"])
    return 160 * n if _is_tough() else 100 * n


def _chain_rosenbrock(x):
    alpha = 4.0
    return float(np.sum((x[:-1] - 1.0) ** 2 + alpha * (x[1:] - x[:-1] ** 2) ** 2))


class StressTests(unittest.TestCase):
    def test_chain_rosenbrock_stress(self):
        algorithm = _selected_algorithm()
        algorithms = (algorithm,) if algorithm else ALGORITHMS
        n = _dimension()
        rng = np.random.default_rng(int(os.environ.get("BDS_TEST_SEED", "2026")))
        x0 = rng.normal(size=n)
        f0 = _chain_rosenbrock(x0)

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm, n=n, tough=_is_tough()):
                result = minimize_bds(
                    _chain_rosenbrock,
                    x0,
                    options={
                        "Algorithm": algorithm,
                        "MaxFunctionEvaluations": _maxfev(n),
                        "StepTolerance": 0.0,
                        "alpha_init": "auto" if algorithm != "ds" else 1.0,
                        "seed": int(os.environ.get("BDS_TEST_SEED", "2026")),
                    },
                )

                self.assertLessEqual(result.nfev, _maxfev(n))
                self.assertTrue(np.isfinite(result.fun))
                self.assertLess(result.fun, f0)


if __name__ == "__main__":
    unittest.main()
