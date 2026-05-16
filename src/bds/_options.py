"""Option handling for BDS.

This module is the Python counterpart of the MATLAB ``set_options`` and
``validate_options`` logic. It keeps the same algorithmic defaults and
validation rules where possible, while also accepting SciPy-style aliases so
``minimize_bds`` can behave like a standard ``scipy.optimize`` solver.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._result import OptimizeWarning
from ._utils import as_positive_vector, is_integer_scalar

_EPS = np.finfo(float).eps


@dataclass
class BDSOptions:
    maxfev: int
    maxiter: int
    ftarget: float
    step_tolerance: np.ndarray
    use_function_value_stop: bool
    func_window_size: int
    func_tol: float
    use_estimated_gradient_stop: bool
    grad_window_size: int
    grad_tol: float
    lipschitz_constant: float
    algorithm: str | None
    direction_set: np.ndarray
    num_blocks: int
    batch_size: int
    replacement_delay: int
    grouped_direction_indices: list[np.ndarray] | None
    block_visiting_pattern: str
    alpha_init: np.ndarray
    expand: float
    shrink: float
    is_noisy: bool
    forcing_function: Callable[[float], float]
    reduction_factor: np.ndarray
    polling_inner: str
    cycling_inner: int
    seed: int | None
    output_xhist: bool
    output_alpha_hist: bool
    output_block_hist: bool
    output_grad_hist: bool
    return_all: bool
    iprint: int
    debug_flag: bool
    gradient_estimation_complete: bool


_OPTION_ALIASES = {
    "MaxFunctionEvaluations": "maxfev",
    "maxfev": "maxfev",
    "maxfun": "maxfev",
    "maxiter": "maxiter",
    "tol": "tol",
    "ftarget": "ftarget",
    "StepTolerance": "step_tolerance",
    "xatol": "step_tolerance",
    "step_tolerance": "step_tolerance",
    "use_function_value_stop": "use_function_value_stop",
    "func_window_size": "func_window_size",
    "func_tol": "func_tol",
    "fatol": "func_tol",
    "ftol": "func_tol",
    "use_estimated_gradient_stop": "use_estimated_gradient_stop",
    "grad_window_size": "grad_window_size",
    "grad_tol": "grad_tol",
    "lipschitz_constant": "lipschitz_constant",
    "Algorithm": "algorithm",
    "algorithm": "algorithm",
    "direction_set": "direction_set",
    "direc": "direction_set",
    "num_blocks": "num_blocks",
    "batch_size": "batch_size",
    "replacement_delay": "replacement_delay",
    "grouped_direction_indices": "grouped_direction_indices",
    "block_visiting_pattern": "block_visiting_pattern",
    "alpha_init": "alpha_init",
    "expand": "expand",
    "shrink": "shrink",
    "is_noisy": "is_noisy",
    "forcing_function": "forcing_function",
    "reduction_factor": "reduction_factor",
    "polling_inner": "polling_inner",
    "cycling_inner": "cycling_inner",
    "seed": "seed",
    "random_state": "seed",
    "output_xhist": "output_xhist",
    "output_alpha_hist": "output_alpha_hist",
    "output_block_hist": "output_block_hist",
    "output_grad_hist": "output_grad_hist",
    "return_all": "return_all",
    "iprint": "iprint",
    "disp": "disp",
    "debug_flag": "debug_flag",
    "gradient_estimation_complete": "gradient_estimation_complete",
}

_IGNORED_DERIVATIVE_KWARGS = {
    "jac",
    "hess",
    "hessp",
}

_DEFAULTS = {
    "maxfev_dim_factor": 500,
    "ftarget": -math.inf,
    "step_tolerance": 1e-6,
    "use_function_value_stop": False,
    "func_window_size": 20,
    "func_tol": 1e-6,
    "use_estimated_gradient_stop": False,
    "grad_window_size": 1,
    "grad_tol": 1e-6,
    "lipschitz_constant": 1e3,
    "block_visiting_pattern": "sorted",
    "alpha_init": 1.0,
    "ds_expand_small": 1.25,
    "ds_shrink_small": 0.4,
    "ds_expand_big": 1.25,
    "ds_shrink_big": 0.4,
    "ds_expand_big_noisy": 1.25,
    "ds_shrink_big_noisy": 0.4,
    "expand_small": 2.0,
    "shrink_small": 0.5,
    "expand_big": 1.25,
    "shrink_big": 0.65,
    "expand_big_noisy": 1.25,
    "shrink_big_noisy": 0.85,
    "is_noisy": False,
    "forcing_function": lambda alpha: alpha**2,
    "reduction_factor": np.array([0.0, _EPS, _EPS], dtype=float),
    "polling_inner": "opportunistic",
    "cycling_inner": 1,
    "seed": None,
    "output_xhist": False,
    "output_alpha_hist": False,
    "output_block_hist": False,
    "output_grad_hist": False,
    "return_all": False,
    "iprint": 0,
    "debug_flag": False,
    "gradient_estimation_complete": False,
}


def canonicalize_options(options: dict | None, extra_options: dict | None = None) -> dict:
    """Merge option dictionaries and map public aliases to internal names.

    MATLAB names such as ``MaxFunctionEvaluations`` and SciPy names such as
    ``maxfev`` are both accepted. Derivative keywords are accepted only for
    SciPy custom-minimizer compatibility; BDS does not use them because it is
    derivative-free. Bound and constraint keywords are accepted only when empty,
    because the BDS problem class is strictly unconstrained.
    """

    merged = {}
    if options:
        merged.update(options)
    if extra_options:
        merged.update({k: v for k, v in extra_options.items() if v is not None})

    canonical = {}
    unknown = []
    for key, value in merged.items():
        if key in _IGNORED_DERIVATIVE_KWARGS:
            if value is not None:
                warnings.warn(
                    f"BDS is derivative-free; option {key!r} is accepted "
                    "for scipy.optimize.minimize compatibility but is not used.",
                    OptimizeWarning,
                    stacklevel=2,
                )
            continue
        if key == "bounds":
            if _has_nonempty_bounds(value):
                raise ValueError("BDS solves unconstrained problems and does not support bounds.")
            continue
        if key == "constraints":
            if _has_nonempty_constraints(value):
                raise ValueError(
                    "BDS solves unconstrained problems and does not support constraints."
                )
            continue
        if key not in _OPTION_ALIASES:
            unknown.append(key)
            continue
        canonical_key = _OPTION_ALIASES[key]
        if canonical_key == "disp":
            canonical["iprint"] = 1 if value else 0
        else:
            canonical[canonical_key] = value
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        warnings.warn(f"Unknown BDS option(s) ignored: {names}.", OptimizeWarning, stacklevel=2)
    return canonical


def make_options(options: dict, n: int, x0: np.ndarray) -> BDSOptions:
    """Validate user options and fill context-dependent defaults."""

    options = dict(options)
    tol = options.pop("tol", None)
    if tol is not None:
        # SciPy's top-level ``tol`` is a generic tolerance. For BDS we interpret
        # it as a shared step, objective-change, and gradient tolerance unless a
        # more specific option has already been supplied.
        tol = _positive_float_option(tol, "tol")
        options.setdefault("step_tolerance", tol)
        options.setdefault("func_tol", tol)
        options.setdefault("grad_tol", tol)

    maxfev = options.pop("maxfev", None)
    if maxfev is None:
        maxfev = _DEFAULTS["maxfev_dim_factor"] * n
    _validate_positive_integer(maxfev, "maxfev")
    maxfev = int(maxfev)
    maxiter = options.pop("maxiter", maxfev)
    _validate_positive_integer(maxiter, "maxiter")
    maxiter = int(maxiter)

    algorithm = options.pop("algorithm", None)
    if algorithm is not None:
        algorithm = str(algorithm).lower()
        if algorithm not in {"cbds", "pbds", "rbds", "pads", "ds"}:
            raise ValueError("algorithm must be one of 'cbds', 'pbds', 'rbds', 'pads', 'ds'.")
        if any(name in options for name in ("block_visiting_pattern", "num_blocks", "batch_size")):
            warnings.warn(
                "algorithm overrides block_visiting_pattern, num_blocks, and batch_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            options.pop("block_visiting_pattern", None)
            options.pop("num_blocks", None)
            options.pop("batch_size", None)

    # Algorithm presets deliberately override the low-level block controls,
    # matching the MATLAB priority rule. Users who want custom block behavior
    # should omit ``algorithm`` and set num_blocks/batch_size/pattern directly.
    if algorithm == "cbds":
        num_blocks = n
        batch_size = n
        block_visiting_pattern = "sorted"
    elif algorithm == "pbds":
        num_blocks = n
        batch_size = n
        block_visiting_pattern = "random"
    elif algorithm == "rbds":
        num_blocks = n
        batch_size = 1
        block_visiting_pattern = "random"
    elif algorithm == "pads":
        num_blocks = n
        batch_size = n
        block_visiting_pattern = "parallel"
    elif algorithm == "ds":
        num_blocks = 1
        batch_size = 1
        block_visiting_pattern = _DEFAULTS["block_visiting_pattern"]
    else:
        num_blocks = options.pop("num_blocks", n)
        _validate_positive_integer(num_blocks, "num_blocks")
        num_blocks = int(num_blocks)
        block_visiting_pattern = str(
            options.pop("block_visiting_pattern", _DEFAULTS["block_visiting_pattern"])
        ).lower()
        batch_size = options.pop("batch_size", num_blocks)
        _validate_positive_integer(batch_size, "batch_size")
        batch_size = int(batch_size)

    if num_blocks > n:
        raise ValueError("num_blocks cannot exceed len(x0).")
    if batch_size > num_blocks:
        raise ValueError("batch_size cannot exceed num_blocks.")
    if block_visiting_pattern not in {"sorted", "random", "parallel"}:
        raise ValueError("block_visiting_pattern must be 'sorted', 'random', or 'parallel'.")

    direction_set = np.asarray(options.pop("direction_set", np.eye(n)), dtype=float)
    if direction_set.shape != (n, n):
        raise ValueError("direction_set must be an n-by-n matrix.")

    step_tolerance = as_positive_vector(
        options.pop("step_tolerance", _DEFAULTS["step_tolerance"]),
        num_blocks,
        name="step_tolerance",
        allow_zero=True,
    )

    replacement_delay = options.pop("replacement_delay", None)
    max_delay = math.floor(num_blocks / batch_size) - 1
    if replacement_delay is None:
        # The default is the largest valid delay, which prioritizes spreading
        # visits across blocks before revisiting the same one.
        replacement_delay = max_delay
    _validate_nonnegative_integer(replacement_delay, "replacement_delay")
    replacement_delay = int(replacement_delay)
    if replacement_delay > max_delay:
        raise ValueError("replacement_delay cannot exceed floor(num_blocks / batch_size) - 1.")

    grouped_direction_indices = options.pop("grouped_direction_indices", None)
    if grouped_direction_indices is not None:
        grouped_direction_indices = _validate_grouped_indices(grouped_direction_indices, n, num_blocks)

    alpha_init = _make_alpha_init(options.pop("alpha_init", _DEFAULTS["alpha_init"]), x0, step_tolerance)
    if alpha_init.size != num_blocks:
        raise ValueError("alpha_init must be a scalar or have length num_blocks.")

    is_noisy = bool(options.pop("is_noisy", _DEFAULTS["is_noisy"]))
    default_expand, default_shrink = _default_expand_shrink(n, algorithm, is_noisy)
    expand = float(options.pop("expand", default_expand))
    shrink = float(options.pop("shrink", default_shrink))
    if not expand >= 1.0:
        raise ValueError("expand must be >= 1.")
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must be in (0, 1).")

    forcing_function = options.pop("forcing_function", _DEFAULTS["forcing_function"])
    if not callable(forcing_function):
        raise ValueError("forcing_function must be callable.")
    test_value = forcing_function(1.0)
    if np.asarray(test_value).shape != ():
        raise ValueError("forcing_function must return a scalar for scalar input.")

    reduction_factor = np.asarray(options.pop("reduction_factor", _DEFAULTS["reduction_factor"]), dtype=float)
    reduction_factor = np.ravel(reduction_factor)
    if reduction_factor.size != 3:
        raise ValueError("reduction_factor must be a length-3 vector.")
    if not (
        reduction_factor[0] <= reduction_factor[1] <= reduction_factor[2]
        and reduction_factor[0] >= 0
        and reduction_factor[1] > 0
    ):
        raise ValueError(
            "reduction_factor must satisfy r0 <= r1 <= r2, r0 >= 0, and r1 > 0."
        )

    polling_inner = str(options.pop("polling_inner", _DEFAULTS["polling_inner"])).lower()
    if polling_inner not in {"opportunistic", "complete"}:
        raise ValueError("polling_inner must be 'opportunistic' or 'complete'.")

    cycling_inner = options.pop("cycling_inner", _DEFAULTS["cycling_inner"])
    _validate_nonnegative_integer(cycling_inner, "cycling_inner")
    cycling_inner = int(cycling_inner)
    if cycling_inner > 3:
        raise ValueError("cycling_inner must be in {0, 1, 2, 3}.")

    seed = options.pop("seed", _DEFAULTS["seed"])
    if seed in (None, "shuffle"):
        seed = None
    else:
        _validate_nonnegative_integer(seed, "seed")
        if int(seed) > 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1].")
        seed = int(seed)

    result = BDSOptions(
        maxfev=maxfev,
        maxiter=maxiter,
        ftarget=float(options.pop("ftarget", _DEFAULTS["ftarget"])),
        step_tolerance=step_tolerance,
        use_function_value_stop=bool(
            options.pop("use_function_value_stop", _DEFAULTS["use_function_value_stop"])
        ),
        func_window_size=_positive_int_option(
            options.pop("func_window_size", _DEFAULTS["func_window_size"]), "func_window_size"
        ),
        func_tol=_positive_float_option(options.pop("func_tol", _DEFAULTS["func_tol"]), "func_tol"),
        use_estimated_gradient_stop=bool(
            options.pop("use_estimated_gradient_stop", _DEFAULTS["use_estimated_gradient_stop"])
        ),
        grad_window_size=_positive_int_option(
            options.pop("grad_window_size", _DEFAULTS["grad_window_size"]), "grad_window_size"
        ),
        grad_tol=_positive_float_option(options.pop("grad_tol", _DEFAULTS["grad_tol"]), "grad_tol"),
        lipschitz_constant=_positive_float_option(
            options.pop("lipschitz_constant", _DEFAULTS["lipschitz_constant"]), "lipschitz_constant"
        ),
        algorithm=algorithm,
        direction_set=direction_set,
        num_blocks=num_blocks,
        batch_size=batch_size,
        replacement_delay=replacement_delay,
        grouped_direction_indices=grouped_direction_indices,
        block_visiting_pattern=block_visiting_pattern,
        alpha_init=alpha_init,
        expand=expand,
        shrink=shrink,
        is_noisy=is_noisy,
        forcing_function=forcing_function,
        reduction_factor=reduction_factor,
        polling_inner=polling_inner,
        cycling_inner=cycling_inner,
        seed=seed,
        output_xhist=bool(options.pop("output_xhist", _DEFAULTS["output_xhist"])),
        output_alpha_hist=bool(options.pop("output_alpha_hist", _DEFAULTS["output_alpha_hist"])),
        output_block_hist=bool(options.pop("output_block_hist", _DEFAULTS["output_block_hist"])),
        output_grad_hist=bool(options.pop("output_grad_hist", _DEFAULTS["output_grad_hist"])),
        return_all=bool(options.pop("return_all", _DEFAULTS["return_all"])),
        iprint=_iprint_option(options.pop("iprint", _DEFAULTS["iprint"])),
        debug_flag=bool(options.pop("debug_flag", _DEFAULTS["debug_flag"])),
        gradient_estimation_complete=bool(
            options.pop("gradient_estimation_complete", _DEFAULTS["gradient_estimation_complete"])
        ),
    )

    if options:
        names = ", ".join(sorted(options))
        warnings.warn(f"Unhandled BDS option(s) ignored: {names}.", OptimizeWarning, stacklevel=2)
    return result


def _has_nonempty_bounds(value) -> bool:
    if value is None:
        return False
    try:
        bounds = list(value)
    except TypeError:
        return True
    return any(bound is not None for bound in bounds)


def _has_nonempty_constraints(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _make_alpha_init(value, x0: np.ndarray, step_tolerance: np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError("alpha_init string value must be 'auto'.")
        if step_tolerance.size != x0.size:
            raise ValueError("alpha_init='auto' requires num_blocks == len(x0).")
        # Smart alpha follows the MATLAB rule: derive coordinate scales from x0
        # while regularizing by StepTolerance. Exact zeros start with step 1;
        # small coordinates keep their local scale; large coordinates are either
        # kept linear or damped logarithmically when the scales of x0 are highly
        # heterogeneous.
        abs_x0 = np.abs(x0)
        regularized_abs_x0 = np.maximum(abs_x0, np.maximum(step_tolerance, np.finfo(float).eps))
        scale_ratio = np.max(regularized_abs_x0) / np.min(regularized_abs_x0)
        if scale_ratio <= 1e2:
            lambda_tau = 0.0
        elif scale_ratio >= 1e3:
            lambda_tau = 1.0
        else:
            lambda_tau = math.log10(scale_ratio) - 2.0

        alpha = np.zeros_like(x0, dtype=float)
        for i, abs_x0_i in enumerate(abs_x0):
            tau_i = step_tolerance[i]
            if tau_i <= 0:
                tau_i = np.finfo(float).eps
            if abs_x0_i <= tau_i:
                alpha[i] = 1.0 - ((1.0 - tau_i) / tau_i) * abs_x0_i
            elif abs_x0_i <= 1.0:
                alpha[i] = abs_x0_i
            else:
                alpha[i] = (1.0 - lambda_tau) * abs_x0_i + lambda_tau * (1.0 + math.log10(abs_x0_i))
        return alpha
    return as_positive_vector(value, step_tolerance.size, name="alpha_init")


def _default_expand_shrink(n: int, algorithm: str | None, is_noisy: bool) -> tuple[float, float]:
    """Return dimension/noise/algorithm dependent step-size factors."""

    is_ds = algorithm == "ds" or n == 1
    if is_ds:
        if n <= 5:
            return _DEFAULTS["ds_expand_small"], _DEFAULTS["ds_shrink_small"]
        if is_noisy:
            return _DEFAULTS["ds_expand_big_noisy"], _DEFAULTS["ds_shrink_big_noisy"]
        return _DEFAULTS["ds_expand_big"], _DEFAULTS["ds_shrink_big"]
    if n <= 5:
        return _DEFAULTS["expand_small"], _DEFAULTS["shrink_small"]
    if is_noisy:
        return _DEFAULTS["expand_big_noisy"], _DEFAULTS["shrink_big_noisy"]
    return _DEFAULTS["expand_big"], _DEFAULTS["shrink_big"]


def _validate_grouped_indices(groups, n: int, num_blocks: int) -> list[np.ndarray]:
    if len(groups) != num_blocks:
        raise ValueError("grouped_direction_indices must have length num_blocks.")
    result = []
    seen = []
    for group in groups:
        arr = np.ravel(np.asarray(group))
        if arr.size == 0:
            raise ValueError("Each group in grouped_direction_indices must be nonempty.")
        if not np.all([is_integer_scalar(value) for value in arr]):
            raise ValueError("grouped_direction_indices must contain integer dimension indices.")
        arr = arr.astype(int)
        if np.any(arr < 1) or np.any(arr > n):
            raise ValueError("grouped_direction_indices entries must be in 1..n.")
        if np.unique(arr).size != arr.size:
            raise ValueError("Each group in grouped_direction_indices must contain unique indices.")
        result.append(arr)
        seen.extend(arr.tolist())
    if sorted(seen) != list(range(1, n + 1)):
        raise ValueError("grouped_direction_indices must partition the dimensions 1..n.")
    return result


def _validate_positive_integer(value, name: str) -> None:
    if not is_integer_scalar(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_nonnegative_integer(value, name: str) -> None:
    if not is_integer_scalar(value) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _positive_int_option(value, name: str) -> int:
    _validate_positive_integer(value, name)
    return int(value)


def _positive_float_option(value, name: str) -> float:
    value = float(value)
    if not value > 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _iprint_option(value) -> int:
    _validate_nonnegative_integer(value, "iprint")
    value = int(value)
    if value > 3:
        raise ValueError("iprint must be in {0, 1, 2, 3}.")
    return value
