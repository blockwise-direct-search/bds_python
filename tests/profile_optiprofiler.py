#!/usr/bin/env python3
"""Run OptiProfiler benchmarks for the Python implementation of BDS.

This module is a profiling/benchmark driver, not a unit test.  It mirrors the
role of the MATLAB ``profile_optiprofiler.m`` script: it translates a small
set of solver names and feature names into an OptiProfiler benchmark call.

Only unconstrained problems are selected.  The Python driver supports the
``plain``, ``noisy``, ``linearly_transformed``, and
``linearly_transformed_noisy`` feature aliases.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from collections.abc import Mapping
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


_ALLOWED_SOLVER_NAMES = {
    "bds",
    "competitor-bds",
    "evolved-bds",
    "evolved-bds-lean",
    "nomad",
    "cbds",
    "pbds",
    "rbds",
    "pads",
    "ds",
    "nelder-mead",
    "powell",
    "cobyla",
    "cobyqa",
}

_BDS_SOLVER_NAMES = {"cbds", "pbds", "rbds", "pads", "ds"}

_COMPETITOR_SOLVER_MODULES = {
    "bds": "competitors.bds",
    "competitor-bds": "competitors.bds",
    "evolved-bds": "competitors.evolved_bds_solver",
    "evolved-bds-lean": "competitors.evolved_bds_solver_lean",
    "nomad": "competitors.nomad_solver",
}

_ALLOWED_PROBLEM_LIBRARIES = {"s2mpj", "pycutest"}

_DEFAULT_EXCLUDELISTS = {
    "s2mpj": {
        "DIAMON2DLS",
        "DIAMON2D",
        "DIAMON3DLS",
        "DIAMON3D",
        "DMN15102LS",
        "DMN15102",
        "DMN15103LS",
        "DMN15103",
        "DMN15332LS",
        "DMN15332",
        "DMN15333LS",
        "DMN15333",
        "DMN37142LS",
        "DMN37142",
        "DMN37143LS",
        "DMN37143",
        "ROSSIMP3_mp",
        "BAmL1SPLS",
        "FBRAIN3LS",
        "GAUSS1LS",
        "GAUSS2LS",
        "GAUSS3LS",
        "HYDC20LS",
        "HYDCAR6LS",
        "LUKSAN11LS",
        "LUKSAN12LS",
        "LUKSAN13LS",
        "LUKSAN14LS",
        "LUKSAN17LS",
        "LUKSAN21LS",
        "LUKSAN22LS",
        "METHANB8LS",
        "METHANL8LS",
        "SPINLS",
        "VESUVIALS",
        "VESUVIOLS",
        "VESUVIOULS",
        "YATP1CLS",
        "MISRA1ALS",
        "OSBORNEA",
        "ECKERLE4LS",
        "NELSONLS",
    },
    "pycutest": {
        "ARGTRIGLS",
        "BROWNAL",
        "COATING",
        "DIAMON2DLS",
        "DIAMON3DLS",
        "DMN15102LS",
        "DMN15103LS",
        "DMN15332LS",
        "DMN15333LS",
        "DMN37142LS",
        "DMN37143LS",
        "ERRINRSM",
        "HYDC20LS",
        "LRA9A",
        "LRCOVTYPE",
        "LUKSAN12LS",
        "LUKSAN14LS",
        "LUKSAN17LS",
        "LUKSAN21LS",
        "LUKSAN22LS",
        "MANCINO",
        "PENALTY2",
        "PENALTY3",
        "VARDIM",
        "GAUSS1LS",
        "GAUSS2LS",
        "GAUSS3LS",
        "CERI651ALS",
        "CERI651BLS",
        "CERI651CLS",
        "CERI651DLS",
        "CERI651ELS",
        "MISRA1ALS",
        "OSBORNEA",
        "ECKERLE4LS",
        "NELSONLS",
    },
}

# PyCUTEst's default metadata puts these problems in the 6--20 dimensional
# range, but the loaded CUTEst instances used by OptiProfiler are actually
# about 500 dimensional.  They are excluded automatically for low-dimensional
# PyCUTEst profiles so BDS is not benchmarked on problems outside the requested
# dimension range.
_PYCUTEST_LARGE_DEFAULT_INSTANCES = {"INTEQNELS", "OSCIPATH"}


def profile_optiprofiler(options: Mapping[str, Any] | SimpleNamespace):
    """Run an OptiProfiler benchmark for selected solvers.

    Parameters
    ----------
    options : mapping or object with attributes
        Benchmark options.  The required keys are ``feature_name`` and
        ``solver_names``.  The solver names must contain at least two entries
        chosen from ``bds``, ``evolved-bds``, ``evolved-bds-lean``,
        ``nomad``, ``cbds``, ``pbds``, ``rbds``, ``pads``, ``ds``,
        ``nelder-mead``, ``powell``, ``cobyla``, and ``cobyqa``.

    Returns
    -------
    tuple
        The ``(solver_scores, profile_scores, curves)`` tuple returned by
        :func:`optiprofiler.benchmark`.
    """

    _configure_matplotlib_environment()

    opts = _as_options_dict(options)
    feature_name_raw = _require_string(opts, "feature_name")
    solver_names = _normalize_solver_names(opts.pop("solver_names", None))
    feature_is_noisy = _is_noisy_feature_name(feature_name_raw)
    feature_name, feature_options, feature_display_name = _parse_feature_name(feature_name_raw)
    _expand_dim_shortcut(opts)

    solvers = [_solver_from_name(name, is_noisy=feature_is_noisy) for name in solver_names]
    benchmark_options = _build_benchmark_options(opts, feature_name, feature_options, solver_names)

    from optiprofiler import benchmark

    with _optiprofiler_feature_display_name(feature_name, feature_display_name):
        return benchmark(solvers, **benchmark_options)


def cbds(fun, x0):
    """Solve an unconstrained problem using cyclic BDS."""

    return _run_bds(fun, x0, "cbds", is_noisy=False)


def pbds(fun, x0):
    """Solve an unconstrained problem using randomly permuted BDS."""

    return _run_bds(fun, x0, "pbds", is_noisy=False)


def rbds(fun, x0):
    """Solve an unconstrained problem using randomized BDS."""

    return _run_bds(fun, x0, "rbds", is_noisy=False)


def pads(fun, x0):
    """Solve an unconstrained problem using parallel BDS."""

    return _run_bds(fun, x0, "pads", is_noisy=False)


def ds(fun, x0):
    """Solve an unconstrained problem using classical direct search."""

    return _run_bds(fun, x0, "ds", is_noisy=False)


def scipy_nelder_mead(fun, x0):
    """Solve an unconstrained problem using SciPy's Nelder-Mead method."""

    return _run_scipy_minimize(fun, x0, "Nelder-Mead")


def scipy_powell(fun, x0):
    """Solve an unconstrained problem using SciPy's Powell method."""

    return _run_scipy_minimize(fun, x0, "Powell")


def scipy_cobyla(fun, x0):
    """Solve an unconstrained problem using SciPy's COBYLA method."""

    return _run_scipy_minimize(fun, x0, "COBYLA")


def scipy_cobyqa(fun, x0):
    """Solve an unconstrained problem using SciPy's COBYQA method."""

    return _run_scipy_minimize(fun, x0, "COBYQA")


def _run_bds(fun, x0, algorithm: str, *, is_noisy: bool):
    from bds import minimize_bds

    x0 = np.asarray(x0, dtype=float)
    result = minimize_bds(
        fun,
        x0,
        options={
            "algorithm": algorithm,
            "is_noisy": is_noisy,
            "maxfev": _max_function_evaluations(x0),
            "maxiter": 10**20,
            "StepTolerance": 1e-6,
        },
    )
    return result.x


def _run_bds_by_algorithm(algorithm: str, is_noisy: bool, fun, x0):
    return _run_bds(fun, x0, algorithm, is_noisy=is_noisy)


def _run_scipy_minimize(fun, x0, method: str):
    from scipy.optimize import minimize

    x0 = np.asarray(x0, dtype=float)
    maxfev = _max_function_evaluations(x0)
    options = {
        "disp": False,
        "maxiter": 10**20,
    }
    if method == "Nelder-Mead":
        options.update({"maxfev": maxfev, "xatol": 1e-6, "fatol": np.finfo(float).eps})
    elif method == "Powell":
        options.update({"maxfev": maxfev, "xtol": 1e-6, "ftol": np.finfo(float).eps})
    elif method in {"COBYLA", "COBYQA"}:
        options.update({"maxiter": maxfev, "tol": 1e-6})
    else:
        raise ValueError(f"Unknown SciPy method: {method}.")

    result = minimize(fun, x0, method=method, options=options)
    return result.x


def _max_function_evaluations(x0) -> int:
    return 200 * np.asarray(x0, dtype=float).size


def _as_options_dict(options: Mapping[str, Any] | SimpleNamespace) -> dict[str, Any]:
    if isinstance(options, Mapping):
        return dict(options)
    if hasattr(options, "__dict__"):
        return dict(vars(options))
    raise TypeError("options must be a mapping or an object with attributes.")


def _configure_matplotlib_environment() -> None:
    """Use a headless, writable Matplotlib setup before OptiProfiler imports it."""

    cache_root = Path(__file__).resolve().parent / "testdata" / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def _require_string(options: dict[str, Any], name: str) -> str:
    if name not in options:
        raise ValueError(f"Please provide the {name}.")
    value = options.pop(name)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _normalize_solver_names(value) -> list[str]:
    if value is None:
        raise ValueError("Please provide the solver_names for the solvers.")
    if isinstance(value, str):
        names = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        try:
            names = [str(item).strip().lower() for item in value]
        except TypeError as exc:
            raise TypeError("solver_names must be a sequence of strings.") from exc

    if len(names) < 2:
        raise ValueError("At least two solver names must be provided for comparison.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate solver name(s): {', '.join(duplicates)}.")
    unknown = sorted(set(names) - _ALLOWED_SOLVER_NAMES)
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_SOLVER_NAMES))
        raise ValueError(f"Unknown solver name(s): {', '.join(unknown)}. Allowed names are: {allowed}.")
    return names


def _parse_feature_name(feature_name: str) -> tuple[str, dict[str, Any], str]:
    feature_name = feature_name.strip().lower()
    if feature_name == "plain":
        return "plain", {}, "plain"
    if feature_name == "linearly_transformed":
        return "linearly_transformed", {}, "linearly_transformed"
    if feature_name == "linearly_transformed_noisy":
        return _linearly_transformed_noisy_feature_options(1e-3)
    if feature_name.startswith("linearly_transformed_noisy_"):
        return _linearly_transformed_noisy_feature_options(
            _parse_noise_level(feature_name.removeprefix("linearly_transformed_noisy_"))
        )
    if feature_name == "noisy":
        return "noisy", {"noise_level": 1e-3}, "noisy_1e-3"
    if feature_name.startswith("noisy_"):
        noise_level = _parse_noise_level(feature_name.rsplit("_", 1)[1])
        return "noisy", {"noise_level": noise_level}, f"noisy_{_format_noise_level_for_stamp(noise_level)}"
    raise ValueError(
        "feature_name must be 'plain', 'linearly_transformed', 'noisy', 'noisy_<level>', "
        "'linearly_transformed_noisy', or 'linearly_transformed_noisy_<level>'."
    )


def _linearly_transformed_noisy_feature_options(noise_level: float) -> tuple[str, dict[str, Any], str]:
    noise_stamp = _format_noise_level_for_stamp(noise_level)
    return (
        "custom",
        {
            "mod_affine": partial(_linearly_transformed_affine_modifier, rotated=True, condition_factor=0.0),
            "mod_fun": partial(_mixed_gaussian_noise_fun_modifier, noise_level=noise_level),
            "_feature_stamp": f"linearly_transformed_noisy_{noise_stamp}",
        },
        f"linearly_transformed_noisy_{noise_stamp}",
    )


@contextmanager
def _optiprofiler_feature_display_name(feature_name: str, display_name: str):
    """Temporarily use a profile-level display name in OptiProfiler titles."""

    from optiprofiler.opclasses import Feature

    original_name = Feature.name
    display_name = _escape_matplotlib_text(display_name)

    def displayed_name(self):
        name = original_name.fget(self)
        return display_name if name == feature_name else name

    Feature.name = property(displayed_name)
    try:
        yield
    finally:
        Feature.name = original_name


def _escape_matplotlib_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("_", "\\_")


def _linearly_transformed_affine_modifier(rng, problem, *, rotated: bool, condition_factor: float):
    from scipy.linalg import qr

    if rotated:
        rand_matrix = rng.standard_normal((problem.n, problem.n))
        q, r = qr(rand_matrix)
        q[:, np.diag(r) < 0] *= -1
    else:
        q = np.eye(problem.n)

    log_condition_number = np.sqrt(condition_factor * problem.n / 2)
    power = np.linspace(-log_condition_number / 2, log_condition_number / 2, problem.n)
    affine = np.diag(2**power) @ q.T
    affine_inverse = q @ np.diag(2 ** (-power))
    return affine, np.zeros(problem.n), affine_inverse


def _mixed_gaussian_noise_fun_modifier(x, rng, problem, *, noise_level: float) -> float:
    f = float(problem.fun(x))
    return float(f + max(1.0, abs(f)) * noise_level * rng.standard_normal())


def _is_noisy_feature_name(feature_name: str) -> bool:
    feature_name = feature_name.strip().lower()
    return feature_name == "noisy" or feature_name.startswith("noisy_") or feature_name.startswith(
        "linearly_transformed_noisy"
    )


def _parse_noise_level(value: str) -> float:
    if "e" in value.lower():
        noise_level = float(value)
    else:
        parsed = float(value)
        noise_level = 10.0 ** (-parsed) if parsed >= 1 and parsed.is_integer() else parsed
    if noise_level <= 0:
        raise ValueError("noise_level must be positive.")
    return noise_level


def _format_noise_level_for_stamp(noise_level: float) -> str:
    exponent = -np.log10(noise_level)
    if np.isclose(exponent, round(exponent)):
        return f"1e-{int(round(exponent))}"
    return str(noise_level).replace(".", "_")


def _build_benchmark_options(
    options: dict[str, Any],
    feature_name: str,
    feature_options: dict[str, Any],
    solver_names: list[str],
) -> dict[str, Any]:
    options = dict(options)
    feature_options = dict(feature_options)
    savepath = Path(options.pop("savepath", Path(__file__).resolve().parent / "testdata"))
    savepath.mkdir(parents=True, exist_ok=True)
    solver_verbose = options.pop("solver_verbose", 2)
    problem_libraries = _normalize_problem_libraries(options.pop("plibs", None))
    feature_stamp = options.pop("feature_stamp", None)
    parsed_feature_stamp = feature_options.pop("_feature_stamp", None)
    options = _apply_problem_exclusions(options, problem_libraries)

    if "ptype" in options and options["ptype"] != "u":
        raise ValueError("BDS profiling only supports unconstrained problems, so ptype must be 'u'.")

    n_runs = options.pop("n_runs", None)
    if n_runs is None:
        n_runs = 1 if feature_name == "plain" else 2
    if feature_stamp is None:
        feature_stamp = parsed_feature_stamp or _default_feature_stamp(feature_name, {**options, **feature_options})
    feature_stamp = _append_problem_library_stamp(str(feature_stamp), problem_libraries)

    benchmark_options = {
        **options,
        **feature_options,
        "benchmark_id": options.pop("benchmark_id", "."),
        "feature_name": feature_name,
        "feature_stamp": feature_stamp,
        "n_runs": n_runs,
        "plibs": problem_libraries,
        "solver_names": [_display_name(name) for name in solver_names],
        "ptype": "u",
        "savepath": str(savepath),
        "solver_verbose": solver_verbose,
    }
    return benchmark_options


def _apply_problem_exclusions(options: dict[str, Any], problem_libraries: list[str]) -> dict[str, Any]:
    options = dict(options)
    exclusions = _normalize_problem_names(options.pop("excludelist", []), "excludelist")

    if not options.get("problem_names"):
        for library in problem_libraries:
            exclusions.extend(sorted(_DEFAULT_EXCLUDELISTS.get(library, ())))

    if _should_exclude_pycutest_large_default_instances(options, problem_libraries):
        exclusions = list(dict.fromkeys([*exclusions, *_PYCUTEST_LARGE_DEFAULT_INSTANCES]))

    if exclusions:
        options["excludelist"] = exclusions
    return options


def _should_exclude_pycutest_large_default_instances(options: dict[str, Any], problem_libraries: list[str]) -> bool:
    if "pycutest" not in problem_libraries:
        return False
    if options.get("problem_names"):
        return False
    if len(problem_libraries) != 1:
        return False

    mindim = options.get("mindim", 1)
    maxdim = options.get("maxdim", np.inf)
    return mindim <= 20 and maxdim <= 20


def _normalize_problem_names(value, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or a sequence of strings.") from exc
    return [str(item).strip() for item in values if str(item).strip()]


def _solver_from_name(name: str, *, is_noisy: bool):
    if name in _COMPETITOR_SOLVER_MODULES:
        return _load_competitor_solver(name)
    if name in _BDS_SOLVER_NAMES:
        return partial(_run_bds_by_algorithm, name, is_noisy)
    return {
        "nelder-mead": scipy_nelder_mead,
        "powell": scipy_powell,
        "cobyla": scipy_cobyla,
        "cobyqa": scipy_cobyqa,
    }[name]


@lru_cache(maxsize=None)
def _load_competitor_solver(name: str):
    tests_dir = Path(__file__).resolve().parent
    tests_dir_str = str(tests_dir)
    if tests_dir_str not in sys.path:
        sys.path.insert(0, tests_dir_str)

    module_name = _COMPETITOR_SOLVER_MODULES[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        path = tests_dir / Path(*module_name.split(".")).with_suffix(".py")
        raise ImportError(f"Cannot import competitor solver from {path}.") from exc

    solver = getattr(module, "solver", None)
    if not callable(solver):
        raise AttributeError(f"Competitor module {module_name} must define callable solver(fun, x0).")
    return solver


def _display_name(name: str) -> str:
    if name in {"bds", "competitor-bds"}:
        return "BDS"
    if name == "evolved-bds":
        return "Evolved BDS"
    if name == "evolved-bds-lean":
        return "Lean Evolved BDS"
    if name == "nomad":
        return "NOMAD"
    if name == "nelder-mead":
        return "Nelder-Mead"
    if name in {"cobyla", "cobyqa"}:
        return name.upper()
    return name


def _normalize_problem_libraries(value) -> list[str]:
    if value is None:
        return ["s2mpj"]
    if isinstance(value, str):
        libraries = [value]
    else:
        try:
            libraries = list(value)
        except TypeError as exc:
            raise TypeError("plibs must be a string or a sequence of strings.") from exc

    libraries = [str(item).strip().lower() for item in libraries if str(item).strip()]
    if not libraries:
        raise ValueError("At least one problem library must be provided.")
    unknown = sorted(set(libraries) - _ALLOWED_PROBLEM_LIBRARIES)
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_PROBLEM_LIBRARIES))
        raise ValueError(f"Unknown problem library name(s): {', '.join(unknown)}. Allowed names are: {allowed}.")
    return libraries


def _default_feature_stamp(feature_name: str, options: dict[str, Any]) -> str:
    if feature_name == "plain":
        return "plain"
    if feature_name == "linearly_transformed":
        feature_stamp = "linearly_transformed"
        if options.get("rotated", True):
            feature_stamp = f"{feature_stamp}_rotated"
        condition_factor = options.get("condition_factor", 0)
        if condition_factor != 0:
            feature_stamp = f"{feature_stamp}_cond{condition_factor}"
        return feature_stamp
    if feature_name == "noisy":
        noise_level = options.get("noise_level", 1e-3)
        noise_type = options.get("noise_type", "mixed")
        feature_stamp = f"noisy_{noise_level}_{noise_type}"
        distribution = options.get("distribution", "gaussian")
        if isinstance(distribution, str) and distribution in {"gaussian", "uniform"}:
            feature_stamp = f"{feature_stamp}_{distribution}"
        return feature_stamp
    return feature_name


def _append_problem_library_stamp(feature_stamp: str, problem_libraries: list[str]) -> str:
    library_stamp = "_".join(problem_libraries)
    suffix = f"_{library_stamp}"
    if feature_stamp.endswith(suffix):
        return feature_stamp
    return f"{feature_stamp}{suffix}"


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver_names", nargs="+", help="Two or more solvers to compare.")
    parser.add_argument(
        "--feature-name",
        default="plain",
        help="plain, linearly_transformed, noisy, noisy_<level>, or linearly_transformed_noisy_<level>.",
    )
    parser.add_argument("--dim", choices=["small", "big", "large"], help="Dimension range shortcut.")
    parser.add_argument("--mindim", type=int, help="Minimum problem dimension.")
    parser.add_argument("--maxdim", type=int, help="Maximum problem dimension.")
    parser.add_argument(
        "--plibs",
        nargs="+",
        choices=sorted(_ALLOWED_PROBLEM_LIBRARIES),
        help="Problem libraries passed to OptiProfiler.",
    )
    parser.add_argument("--problem-names", nargs="+", help="Specific problem names to benchmark.")
    parser.add_argument("--excludelist", nargs="+", help="Problem names to exclude from the benchmark.")
    parser.add_argument("--n-runs", type=int, help="Number of runs.")
    parser.add_argument("--n-jobs", type=int, help="Number of OptiProfiler workers.")
    parser.add_argument("--solver-verbose", type=int, choices=[0, 1, 2], help="Solver verbosity.")
    parser.add_argument("--savepath", help="Directory where OptiProfiler stores benchmark results.")
    parser.add_argument("--benchmark-id", help="OptiProfiler benchmark identifier.")
    parser.add_argument("--run-plain", action="store_true", help="Also run plain problems for comparison.")
    return parser.parse_args()


def _options_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "feature_name": args.feature_name,
        "solver_names": args.solver_names,
    }
    for key in (
        "dim",
        "mindim",
        "maxdim",
        "plibs",
        "problem_names",
        "excludelist",
        "n_runs",
        "n_jobs",
        "solver_verbose",
        "savepath",
        "benchmark_id",
    ):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if args.run_plain:
        options["run_plain"] = True
    _expand_dim_shortcut(options)
    return options


def _expand_dim_shortcut(options: dict[str, Any]) -> None:
    dim = options.pop("dim", None)
    if dim is None:
        return
    if "mindim" in options or "maxdim" in options:
        raise ValueError("Do not combine dim with mindim or maxdim.")
    if dim == "small":
        options["mindim"] = 1
        options["maxdim"] = 5
    elif dim == "big":
        options["mindim"] = 6
        options["maxdim"] = 20
    elif dim == "large":
        options["mindim"] = 21
        options["maxdim"] = 200
    else:
        raise ValueError("dim must be 'small', 'big', or 'large'.")


def main() -> None:
    profile_optiprofiler(_options_from_cli(_parse_cli_args()))


if __name__ == "__main__":
    main()
