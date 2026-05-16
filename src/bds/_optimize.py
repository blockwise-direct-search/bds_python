"""SciPy-style public optimizer for Blockwise Direct Search."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._directions import (
    cycling,
    direction_probability_matrix,
    divide_direction_set,
    get_direction_set,
)
from ._exit import (
    CALLBACK_STOP,
    FTARGET_REACHED,
    GRADIENT_ESTIMATION_COMPLETED,
    MAXFUN_REACHED,
    MAXIT_REACHED,
    MESSAGES,
    SMALL_ALPHA,
    SMALL_ESTIMATED_GRADIENT,
    SMALL_OBJECTIVE_CHANGE,
    SUCCESS_STATUSES,
)
from ._gradient import GradientInfo, estimate_gradient, gradient_error_bound
from ._options import BDSOptions, canonicalize_options, make_options
from ._result import OptimizeResult
from ._utils import as_float_array, eval_fun, print_vector


@dataclass
class InnerSearchResult:
    x: np.ndarray
    fun: float
    status: int | None
    fhist: list[float]
    xhist: list[np.ndarray]
    invalid_points: list[np.ndarray]
    nfev: int
    direction_indices: np.ndarray
    terminate: bool
    sufficient_decrease: bool


def minimize_bds(
    fun,
    x0,
    args: tuple = (),
    options: dict[str, Any] | None = None,
    callback=None,
    **unknown_options,
) -> OptimizeResult:
    """Minimize a scalar function using Blockwise Direct Search.

    BDS solves unconstrained optimization problems without derivatives. Starting
    from ``x0``, it repeatedly polls along paired directions
    ``{d_0, -d_0, ..., d_{n-1}, -d_{n-1}}``. These paired directions are divided
    into blocks; each outer iteration selects a batch of blocks and performs a
    classical direct-search step inside each selected block.

    The interface follows ``scipy.optimize.minimize`` conventions. The return
    value is an ``OptimizeResult`` with fields such as ``x``, ``fun``,
    ``success``, ``status``, ``message``, ``nfev``, ``nit``, and ``fhist``.
    MATLAB/BDS option names are accepted where they carry algorithmic meaning,
    while SciPy-style aliases such as ``maxiter``, ``maxfev``, ``xatol``,
    ``fatol``, ``tol``, ``disp``, and ``return_all`` are also supported.

    Important BDS options
    ---------------------
    expand, shrink : float
        Expansion and shrinkage factors for block step sizes. ``expand`` must
        be at least 1 and ``shrink`` must lie in ``(0, 1)``. Defaults depend on
        dimension, ``algorithm``, and ``is_noisy``.
    num_blocks : int
        Number of blocks, no larger than the dimension. By default each
        coordinate pair ``(d_i, -d_i)`` forms its own block.
    direction_set, direc : array_like, shape (n, n)
        Columns define the positive polling directions. The complete polling
        set is built as ``[d_0, -d_0, ..., d_{n-1}, -d_{n-1}]``. Nearly
        dependent direction sets are repaired to obtain a basis.
    alpha_init : float, array_like, or "auto"
        Initial step size. Scalars are applied to every block; vectors specify
        one value per block. ``"auto"`` derives coordinate-wise scales from
        ``x0`` and ``StepTolerance`` and assumes the default coordinate
        direction grouping.
    forcing_function : callable
        Function used in sufficient-decrease tests. The default is
        ``lambda alpha: alpha**2``.
    reduction_factor : array_like, shape (3,)
        Factors used for base-point updates and step-size updates. They must
        satisfy ``r0 <= r1 <= r2``, ``r0 >= 0``, and ``r1 > 0``.

    Advanced BDS options
    --------------------
    algorithm : {"cbds", "pbds", "rbds", "pads", "ds"}
        Preset for common blockwise direct-search variants: sorted BDS,
        randomly permuted BDS, randomized BDS, parallel BDS, and classical
        direct search.
    batch_size : int
        Number of blocks sampled in each outer iteration.
    replacement_delay : int
        Delay before a selected block can be selected again. The default is the
        largest value allowed by ``floor(num_blocks / batch_size) - 1``.
    grouped_direction_indices : sequence of sequences
        User grouping of dimensions. Indices are 1-based for MATLAB
        compatibility at the public option boundary; internally they are
        converted to Python's 0-based direction indices.
    block_visiting_pattern : {"sorted", "random", "parallel"}
        Controls the order in which sampled blocks are processed. In
        ``"parallel"`` mode all blocks share the same base point within an
        outer iteration, and the base point is updated only after the batch.
    polling_inner : {"opportunistic", "complete"}
        Whether to stop polling a block as soon as sufficient decrease is found
        or to evaluate all directions in the block.
    cycling_inner : {0, 1, 2, 3}
        Direction-order cycling strategy used after opportunistic success.

    Termination and output options
    ------------------------------
    ftarget : float
        Stop once the objective value is at most this target.
    StepTolerance, xatol : float or array_like
        Stop when every block step size falls below its threshold.
    use_function_value_stop, func_window_size, func_tol
        Optional stopping rule based on small objective change over a sliding
        window. ``fatol`` and ``ftol`` are aliases for ``func_tol``.
    use_estimated_gradient_stop, grad_window_size, grad_tol, lipschitz_constant
        Optional stopping rule based on the estimated gradient and an error
        bound.
    output_xhist, output_alpha_hist, output_block_hist, output_grad_hist
        Include histories of evaluated points, step sizes, visited blocks, and
        estimated gradients in the result.

    Notes
    -----
    ``jac``, ``hess``, and ``hessp`` are accepted for SciPy custom-minimizer
    compatibility but ignored with an ``OptimizeWarning`` when supplied. Bounds
    and constraints are not part of the BDS problem class; nonempty
    ``bounds``/``constraints`` inputs raise ``ValueError``.
    """

    x0_arr = as_float_array(x0, name="x0")
    original_shape = np.asarray(x0).shape

    if args:

        def wrapped(x):
            return fun(x, *args)

    else:
        wrapped = fun

    canonical = canonicalize_options(options, unknown_options)
    opts = make_options(canonical, x0_arr.size, x0_arr)
    return _minimize_bds(wrapped, x0_arr, original_shape, opts, callback)


def bds(fun, x0, args: tuple = (), options: dict[str, Any] | None = None, callback=None, **kwargs):
    """Alias for :func:`minimize_bds`."""

    return minimize_bds(fun, x0, args=args, options=options, callback=callback, **kwargs)


def _minimize_bds(fun, x0: np.ndarray, original_shape, opts: BDSOptions, callback) -> OptimizeResult:
    """Run the outer blockwise direct-search iteration.

    The notation mirrors the MATLAB implementation:

    * ``xopt``/``fopt`` store the best point and value seen globally.
    * ``xbase``/``fbase`` are the base point and value used to measure
      reduction inside the next block. In non-parallel modes they may update
      after each block; in parallel mode they update only after a full batch.
    * ``alpha_all[i]`` is the current step size for block ``i``.
    """

    n = x0.size
    rng = np.random.default_rng(opts.seed)

    directions = get_direction_set(n, opts)
    positive_direction_set = directions[:, 0::2]
    grouped_direction_indices = divide_direction_set(n, opts.num_blocks, opts)

    alpha_all = opts.alpha_init.astype(float, copy=True)
    alpha_tol = opts.step_tolerance

    # Initialize sliding windows so optional stopping tests stay inactive until
    # enough real values have replaced the sentinels. ``inf`` is used for
    # objective changes because max-min is meaningful only after the window is
    # populated; ``nan`` disables the gradient test until reliable estimates
    # are recorded.
    fopt_window = np.full(opts.func_window_size, np.inf)
    norm_grad_window = np.full(opts.grad_window_size, np.nan)
    record_gradient_norm = False
    reference_grad_norm = np.nan

    xhist: list[np.ndarray] = []
    invalid_points: list[np.ndarray] = []
    fhist: list[float] = []
    alpha_hist: list[np.ndarray] = []
    block_hist: list[int] = []
    grad_hist: list[np.ndarray] = []
    grad_xhist: list[np.ndarray] = []
    grad_iter: list[int] = []
    allvecs: list[np.ndarray] = []

    if opts.output_alpha_hist:
        alpha_hist.append(alpha_all.copy())

    # Gradient estimation uses two views of step sizes: per batch for the
    # finite-difference estimate, and per block for the error bound, which
    # requires information about all blocks rather than only the sampled ones.
    grad_info = GradientInfo(
        n=n,
        complete_direction_set=directions,
        step_size_per_batch=np.full(opts.batch_size, np.nan),
        step_size_per_block=alpha_all.copy(),
        fbase_per_batch=np.full(opts.batch_size, np.nan),
    )

    # The status is initialized to the iteration-limit code. Any expected
    # termination condition below will overwrite it before returning.
    status = MAXIT_REACHED
    evaluation = eval_fun(fun, x0)
    nfev = 1
    nit = 0

    if opts.output_xhist:
        xhist.append(x0.copy())
        if not evaluation.is_valid:
            invalid_points.append(x0.copy())
    fhist.append(evaluation.raw_value)

    if opts.iprint >= 2:
        print("The initial step size is:")
        print_vector(alpha_all)
        print(f"Function number {nfev:d}    F = {evaluation.raw_value:23.16E}")
        print("The corresponding X is:")
        print_vector(x0)

    # At this point only x0 has been evaluated, so it is both the best point and
    # the first base point. Histories store the raw objective value, while the
    # algorithm uses the moderated value returned by ``eval_fun``.
    xopt = x0.copy()
    fopt = evaluation.value
    f0 = evaluation.value
    f0_raw = evaluation.raw_value
    fopt_window = _push_window(fopt_window, fopt)
    if opts.return_all:
        allvecs.append(xopt.copy())

    terminate = False
    if evaluation.raw_value <= opts.ftarget:
        terminate = True
        status = FTARGET_REACHED
    elif nfev >= opts.maxfev:
        terminate = True
        status = MAXFUN_REACHED

    # ``xbase`` is the reference point for sufficient-decrease tests inside the
    # next block. It is intentionally distinct from ``xopt``: a trial point may
    # become globally best only after a block/iteration is completed.
    xbase = xopt.copy()
    fbase = fopt
    fopt_all = np.full(opts.num_blocks, np.nan)
    xopt_all = np.full((n, opts.num_blocks), np.nan)

    for iteration in range(1, opts.maxiter + 1):
        if terminate:
            break
        nit = iteration

        # Blocks visited in the most recent replacement-delay window are made
        # unavailable. This encourages coverage of different blocks when
        # ``batch_size`` is smaller than ``num_blocks``.
        unavailable = _recent_blocks(block_hist, iteration, opts.replacement_delay, opts.batch_size)
        available = np.setdiff1d(np.arange(opts.num_blocks), unavailable, assume_unique=False)
        if available.size < opts.batch_size:
            available = np.arange(opts.num_blocks)

        # First sample the batch, then optionally sort it. Sorting affects visit
        # order, not the sampling probabilities used by the gradient estimator.
        block_indices = rng.choice(available, size=opts.batch_size, replace=False)
        probability = direction_probability_matrix(
            n, opts.batch_size, grouped_direction_indices, available
        )
        grad_info.direction_selection_probability_matrix = probability

        if opts.block_visiting_pattern == "sorted":
            block_indices = np.sort(block_indices)

        sampled_direction_indices_per_batch: list[np.ndarray] = []
        function_values_per_batch: list[np.ndarray] = []
        batch_gradient_available = np.zeros(opts.batch_size, dtype=bool)

        fopt_all[:] = np.nan
        xopt_all[:, :] = np.nan

        for batch_pos, block_idx in enumerate(block_indices):
            direction_indices = grouped_direction_indices[block_idx]
            grad_info.step_size_per_block[block_idx] = alpha_all[block_idx]
            grad_info.step_size_per_batch[batch_pos] = alpha_all[block_idx]
            grad_info.fbase_per_batch[batch_pos] = fbase

            inner = _inner_direct_search(
                fun,
                xbase,
                fbase,
                directions[:, direction_indices],
                direction_indices,
                alpha_all[block_idx],
                opts,
                opts.maxfev - nfev,
                nfev,
                int(block_idx),
            )

            block_hist.append(int(block_idx))

            if opts.output_xhist:
                xhist.extend(point.copy() for point in inner.xhist)
                invalid_points.extend(point.copy() for point in inner.invalid_points)
            fhist.extend(inner.fhist)
            nfev += inner.nfev

            sampled_direction_indices_per_batch.append(direction_indices[: inner.nfev].copy())
            function_values_per_batch.append(np.asarray(inner.fhist, dtype=float))
            batch_gradient_available[batch_pos] = (
                inner.nfev == direction_indices.size and not inner.sufficient_decrease
            )

            fopt_all[block_idx] = inner.fun
            xopt_all[:, block_idx] = inner.x
            grouped_direction_indices[block_idx] = inner.direction_indices

            # Decide whether the block's best point is good enough to become
            # the next base point. This must be checked before changing the
            # step size, because the sufficient-decrease threshold uses the
            # step size from the current block search.
            update_base = _is_sufficient_for_base_update(
                inner.fun,
                fbase,
                opts.reduction_factor[0],
                opts.forcing_function(alpha_all[block_idx]),
            )

            # Expand on a large reduction, shrink on too little reduction, and
            # keep the step size otherwise. The forcing function always sees
            # the current step size, not the next one.
            forcing = opts.forcing_function(alpha_all[block_idx])
            if _is_sufficient_for_base_update(inner.fun, fbase, opts.reduction_factor[2], forcing):
                alpha_all[block_idx] *= opts.expand
            elif inner.fun + opts.reduction_factor[1] * forcing >= fbase or (
                math.isnan(inner.fun) and not math.isnan(fbase)
            ):
                alpha_all[block_idx] *= opts.shrink

            if inner.terminate:
                terminate = True
                status = inner.status if inner.status is not None else MAXFUN_REACHED
                break

            if np.all(alpha_all < alpha_tol):
                terminate = True
                status = SMALL_ALPHA
                break

            if opts.block_visiting_pattern != "parallel" and update_base:
                xbase = inner.x.copy()
                fbase = inner.fun

        if opts.output_alpha_hist:
            alpha_hist.append(alpha_all.copy())

        # Update the global best only after the iteration has collected block
        # results. This avoids letting a later base-point update make
        # ``fopt_all`` inconsistent with the best value already observed.
        finite_mask = ~np.isnan(fopt_all)
        if np.any(finite_mask):
            candidate_indices = np.flatnonzero(finite_mask)
            best_local_index = candidate_indices[np.argmin(fopt_all[finite_mask])]
            if fopt_all[best_local_index] < fopt:
                fopt = float(fopt_all[best_local_index])
                xopt = xopt_all[:, best_local_index].copy()

        fopt_window = _push_window(fopt_window, fopt)

        if opts.block_visiting_pattern == "parallel":
            # In parallel mode every selected block used the same base point.
            # Therefore the base can be updated only once the whole batch has
            # finished, and only if the batch-level reduction is sufficient.
            forcing = opts.forcing_function(float(np.min(alpha_all)))
            if (opts.reduction_factor[0] <= 0 and fopt < fbase) or (
                fopt + opts.reduction_factor[0] * forcing < fbase
            ):
                xbase = xopt.copy()
                fbase = fopt

        if opts.use_function_value_stop:
            # Use ``fopt - f0`` rather than ``fopt`` itself so that this
            # optional test is invariant under adding a constant to the
            # objective function.
            func_change = np.max(fopt_window) - np.min(fopt_window)
            if func_change < opts.func_tol * min(1.0, abs(fopt - f0)) or func_change < (
                1e-3 * opts.func_tol * max(1.0, abs(fopt - f0))
            ):
                terminate = True
                status = SMALL_OBJECTIVE_CHANGE

        if np.all(batch_gradient_available):
            # A block contributes to gradient estimation only when every
            # direction in that block was evaluated and no sufficient decrease
            # was found. Otherwise incomplete directional information could
            # create a misleading finite-difference estimate.
            grad_info.sampled_direction_indices_per_batch = sampled_direction_indices_per_batch
            grad_info.function_values_per_batch = function_values_per_batch
            grad = estimate_gradient(grad_info)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm <= np.sqrt(n) * 1e30 and grad_norm > 10.0 * np.sqrt(n) * np.finfo(float).eps:
                grad_hist.append(grad.copy())
                grad_xhist.append(xbase.copy())
                grad_iter.append(iteration)

                if opts.gradient_estimation_complete and iteration > 1:
                    terminate = True
                    status = GRADIENT_ESTIMATION_COMPLETED

                if opts.use_estimated_gradient_stop:
                    # The gradient error bound needs all block step sizes and
                    # the full block-direction assignment, not just the blocks
                    # sampled in this iteration.
                    grad_error = gradient_error_bound(
                        grad_info.step_size_per_block,
                        opts.batch_size,
                        grouped_direction_indices,
                        n,
                        positive_direction_set,
                        probability,
                        opts.lipschitz_constant,
                    )
                    if not record_gradient_norm:
                        # Start the stopping window only after the first
                        # gradient estimate with a sufficiently small error
                        # bound. From then on, recorded values are conservative
                        # upper bounds ``norm(grad) + grad_error``.
                        if grad_error < max(1e-3, 1e-1 * grad_norm):
                            reference_grad_norm = grad_norm
                            record_gradient_norm = True
                    else:
                        norm_grad_window = _push_window(norm_grad_window, grad_norm + grad_error)

                    if record_gradient_norm and np.all(
                        (norm_grad_window < opts.grad_tol * min(1.0, reference_grad_norm))
                        | (
                            norm_grad_window
                            < 1e-3 * opts.grad_tol * max(1.0, reference_grad_norm)
                        )
                    ):
                        terminate = True
                        status = SMALL_ESTIMATED_GRADIENT

        if callback is not None:
            try:
                _call_callback(callback, xopt, fopt)
            except StopIteration:
                terminate = True
                status = CALLBACK_STOP

        if opts.return_all:
            allvecs.append(xopt.copy())

        if terminate:
            break

    result = OptimizeResult()
    result.x = np.reshape(xopt, original_shape)
    result.fun = float(fopt)
    result.success = status in SUCCESS_STATUSES
    result.status = int(status)
    result.message = MESSAGES.get(status, "Unknown BDS termination status.")
    result.nfev = int(nfev)
    result.nit = int(nit)
    result.fhist = np.asarray(fhist, dtype=float)

    if opts.output_block_hist:
        result.blocks_hist = np.asarray(block_hist, dtype=int) + 1
    if opts.output_alpha_hist:
        result.alpha_hist = np.column_stack(alpha_hist) if alpha_hist else np.empty((opts.num_blocks, 0))
    if opts.output_xhist:
        result.xhist = np.column_stack(xhist) if xhist else np.empty((n, 0))
        result.invalid_points = (
            np.column_stack(invalid_points) if invalid_points else np.empty((n, 0))
        )
    if opts.return_all:
        result.allvecs = [point.reshape(original_shape) for point in allvecs]
    if opts.output_grad_hist:
        result.grad_hist = np.column_stack(grad_hist) if grad_hist else np.empty((n, 0))
        result.grad_xhist = np.column_stack(grad_xhist) if grad_xhist else np.empty((n, 0))
        result.grad_iter = np.asarray(grad_iter, dtype=int)

    result.final_simplex = None
    result.maxcv = 0.0

    if opts.iprint >= 1:
        print()
        print(result.message)
        print(f"Number of function values = {nfev:d}    Least value of F is {fopt:23.16E}")
        print("The corresponding X is:")
        print_vector(result.x)

    if opts.debug_flag:
        _verify_postconditions(result, f0_raw)

    return result


def _inner_direct_search(
    fun,
    xbase: np.ndarray,
    fbase: float,
    directions: np.ndarray,
    direction_indices: np.ndarray,
    alpha: float,
    opts: BDSOptions,
    maxfev_remaining: int,
    nfev_exhausted: int,
    block_idx: int,
) -> InnerSearchResult:
    """Perform one classical direct-search pass inside a block.

    ``directions`` contains only the directions assigned to the current block,
    while ``direction_indices`` stores their indices in the complete direction
    set. Function histories keep the raw value returned by the user objective;
    comparisons use the moderated value from ``eval_fun`` so that failed or NaN
    evaluations behave as bad trial points without erasing diagnostic history.
    """

    fopt = fbase
    xopt = xbase.copy()
    fhist: list[float] = []
    xhist: list[np.ndarray] = []
    invalid_points: list[np.ndarray] = []
    sufficient_decrease = False
    status = None

    # At most one pass over the block directions is made. The remaining global
    # function-evaluation budget may cut this pass short.
    max_evaluations = max(0, min(maxfev_remaining, directions.shape[1]))
    nfev = 0

    for j in range(max_evaluations):
        # Evaluate the current polling direction from the block base point.
        xnew = xbase + alpha * directions[:, j]
        evaluation = eval_fun(fun, xnew)
        nfev += 1
        fhist.append(evaluation.raw_value)
        xhist.append(xnew.copy())
        if not evaluation.is_valid:
            invalid_points.append(xnew.copy())

        if opts.iprint >= 2:
            print(f"The {block_idx + 1:d}-th block is currently being visited.")
            print("The corresponding step size is:")
            print(f"{alpha:23.16E}")
            print(f"Function number {nfev_exhausted + nfev:d}    F = {evaluation.raw_value:23.16E}")
            print("The corresponding X is:")
            print_vector(xnew)

        # Defensive NaN handling mirrors the MATLAB code: eval_fun normally
        # turns NaN into inf for comparisons, but if this policy changes later,
        # any non-NaN value should still improve over a NaN incumbent.
        if evaluation.value < fopt or (math.isnan(fopt) and not math.isnan(evaluation.value)):
            xopt = xnew.copy()
            fopt = evaluation.value

        # ftarget and maxfev are checked immediately after every evaluation so
        # the inner search does not spend budget after a hard stop is reached.
        if evaluation.value <= opts.ftarget or nfev >= maxfev_remaining:
            break

        # Sufficient decrease serves two purposes: opportunistic polling stops
        # early when it is achieved, and gradient estimation is allowed only for
        # blocks where it was not achieved after complete polling.
        sufficient_decrease = (
            evaluation.value + opts.reduction_factor[0] * opts.forcing_function(alpha) / 2.0 < fbase
        ) or (math.isnan(fbase) and not math.isnan(evaluation.value))

        if sufficient_decrease and opts.polling_inner != "complete":
            # Cycling preserves useful direction-order memory for the next time
            # this block is visited by the outer loop.
            direction_indices = cycling(direction_indices, j, opts.cycling_inner)
            break

    terminate = False
    if fhist:
        # Among inner-loop exits, reaching ftarget has priority over exhausting
        # the evaluation budget. Otherwise the outer loop continues with other
        # stopping criteria.
        last_raw = fhist[-1]
        last_value = math.inf if math.isnan(last_raw) else last_raw
        terminate = last_value <= opts.ftarget or nfev >= maxfev_remaining
        if last_value <= opts.ftarget:
            status = FTARGET_REACHED
        elif nfev >= maxfev_remaining:
            status = MAXFUN_REACHED
    elif maxfev_remaining <= 0:
        terminate = True
        status = MAXFUN_REACHED

    if opts.iprint >= 3:
        if sufficient_decrease:
            print(f"Sufficient decrease achieved in the {block_idx + 1:d}-th block.")
        else:
            print(f"Sufficient decrease not achieved in the {block_idx + 1:d}-th block.")
        print(f"The decrease value of the {block_idx + 1:d}-th block is:{fopt - fbase:23.16E}")
        print()

    return InnerSearchResult(
        x=xopt,
        fun=float(fopt),
        status=status,
        fhist=fhist,
        xhist=xhist,
        invalid_points=invalid_points,
        nfev=nfev,
        direction_indices=direction_indices,
        terminate=terminate,
        sufficient_decrease=sufficient_decrease,
    )


def _recent_blocks(
    block_hist: list[int],
    iteration: int,
    replacement_delay: int,
    batch_size: int,
) -> np.ndarray:
    if replacement_delay <= 0 or not block_hist:
        return np.empty(0, dtype=int)
    start = max(0, (iteration - replacement_delay - 1) * batch_size)
    recent = block_hist[start:]
    return np.unique(np.asarray(recent, dtype=int))


def _is_sufficient_for_base_update(
    candidate: float,
    base: float,
    factor: float,
    forcing_value: float,
) -> bool:
    return (candidate + factor * forcing_value < base) or (math.isnan(base) and not math.isnan(candidate))


def _push_window(window: np.ndarray, value: float) -> np.ndarray:
    updated = np.empty_like(window)
    updated[:-1] = window[1:]
    updated[-1] = value
    return updated


def _call_callback(callback, x: np.ndarray, fun: float) -> None:
    signature = inspect.signature(callback)
    if "intermediate_result" in signature.parameters:
        intermediate_result = OptimizeResult(x=np.array(x, copy=True), fun=float(fun))
        callback(intermediate_result=intermediate_result)
    else:
        callback(np.array(x, copy=True))


def _verify_postconditions(result: OptimizeResult, f0_raw: float) -> None:
    if result.nfev != result.fhist.size:
        raise RuntimeError("BDS internal error: nfev does not match fhist length.")
    if np.isfinite(result.fun) and np.isfinite(f0_raw) and result.fun > f0_raw + 1e-12:
        raise RuntimeError("BDS internal error: final value is worse than the initial value.")
