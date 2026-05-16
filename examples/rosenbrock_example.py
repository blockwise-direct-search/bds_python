"""Run BDS on the Rosenbrock function."""

from __future__ import annotations

import numpy as np

from bds import minimize_bds


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


if __name__ == "__main__":
    result = minimize_bds(
        rosenbrock,
        np.array([-1.2, 1.0]),
        options={
            "Algorithm": "cbds",
            "MaxFunctionEvaluations": 10000,
            "StepTolerance": 1e-5,
            "alpha_init": "auto",
            "seed": 0,
        },
    )
    print(result)
