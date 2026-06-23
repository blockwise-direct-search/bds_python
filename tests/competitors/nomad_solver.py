"""NOMAD competitor wrapper for OptiProfiler.

OptiProfiler expects ``solver(fun, x0)``.  This wrapper uses the PyNomad
callback interface and returns NOMAD's best point under the same evaluation
budget used by the Lean Evolved BDS smoke tests.
"""
from __future__ import annotations

import math

import numpy as np


MAX_EVAL_FACTOR = 200


def _format_value(value: float) -> bytes:
    if not math.isfinite(value):
        value = np.finfo(float).max
    return repr(float(value)).encode("utf-8")


def solver(fun, x0):
    try:
        import PyNomad
    except ImportError as exc:
        raise ImportError("NOMAD competitor requires the PyNomadBBO package.") from exc

    x0 = np.asarray(x0, dtype=float).ravel()
    n = int(x0.size)
    max_eval = max(1, MAX_EVAL_FACTOR * n)

    best_x = x0.copy()
    best_f = float("inf")

    def blackbox(eval_point):
        nonlocal best_x, best_f

        x = np.array([eval_point.get_coord(i) for i in range(eval_point.size())], dtype=float)
        try:
            f = float(fun(x))
        except Exception:
            eval_point.setBBO(_format_value(float("inf")))
            return False

        if math.isfinite(f) and f < best_f:
            best_f = f
            best_x = x.copy()

        eval_point.setBBO(_format_value(f))
        return True

    params = [
        "BB_OUTPUT_TYPE OBJ",
        f"MAX_BB_EVAL {max_eval}",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "DISPLAY_STATS BBE OBJ",
    ]

    result = PyNomad.optimize(blackbox, x0.tolist(), [], [], params)
    x_best = result.get("x_best") if isinstance(result, dict) else None
    if x_best:
        return np.asarray(x_best, dtype=float)
    return best_x.copy()
