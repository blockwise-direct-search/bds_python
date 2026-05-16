"""Internal utility functions."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class FunctionEvaluation:
    value: float
    raw_value: float
    is_valid: bool


def as_float_array(x, *, name: str) -> np.ndarray:
    """Convert input to a finite-dimensional 1-D float array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a one-dimensional array.")
    arr = np.ravel(arr).astype(float, copy=True)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(np.iscomplex(arr)):
        raise ValueError(f"{name} must be real-valued.")
    return arr


def is_integer_scalar(value) -> bool:
    if isinstance(value, bool):
        return False
    return np.isscalar(value) and float(value).is_integer()


def as_positive_vector(value, length: int, *, name: str, allow_zero: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr))
    else:
        arr = np.ravel(arr).astype(float, copy=True)
        if arr.size != length:
            raise ValueError(f"{name} must be a scalar or have length {length}.")
    if allow_zero:
        ok = np.all(arr >= 0)
        condition = "nonnegative"
    else:
        ok = np.all(arr > 0)
        condition = "positive"
    if not ok:
        raise ValueError(f"{name} must be {condition}.")
    return arr


def eval_fun(fun: Callable[[np.ndarray], float], x: np.ndarray) -> FunctionEvaluation:
    """Evaluate the objective and normalize failures for the algorithm.

    BDS distinguishes the raw function value from the value used internally.
    Histories store the raw value so users can see NaNs and failed evaluations;
    comparisons use ``inf`` for NaN or failed evaluations so those trial points
    are treated as bad but the search can continue. Infinite and very large
    finite values are left unchanged.
    """

    is_valid = True
    try:
        raw = fun(np.array(x, copy=True))
    except Exception:
        warnings.warn("The function evaluation failed.", RuntimeWarning, stacklevel=2)
        return FunctionEvaluation(value=math.inf, raw_value=math.nan, is_valid=False)

    raw_arr = np.asarray(raw)
    if raw_arr.shape != ():
        raise ValueError("The objective function must return a scalar.")

    raw_value = float(raw_arr)
    value = raw_value
    if math.isnan(raw_value):
        value = math.inf
        is_valid = False
    return FunctionEvaluation(value=value, raw_value=raw_value, is_valid=is_valid)


def print_vector(x: np.ndarray) -> None:
    print(" ".join(f"{value:23.16E}" for value in np.ravel(x)))
