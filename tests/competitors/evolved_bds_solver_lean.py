"""Lean Evolved BDS with the most promising non-coordinate strategies.

This variant keeps the strategies that looked useful in the local
investigation:
- explicit productive displacement memory beyond cycling
- sweep-level pattern / momentum extrapolation
- low-dimensional diagonal probing on stagnation

It deliberately drops the mixed heuristics from the full evolved solver:
- success-rate coordinate ordering
- smoothed sign preference beyond ordinary direction cycling
- per-coordinate immediate extension
- step-size recovery

OptiProfiler expects ``solver(fun, x0)``. NumPy only.
"""
from __future__ import annotations

import numpy as np


MAX_EVAL_FACTOR = 200


def inner_direct_search(fun, xbase, fbase, D, direction_indices, alpha, submaxfun):
    """Poll in given directions from xbase, stop on improvement or budget."""

    direction_indices = list(direction_indices)
    exitflag = np.nan
    terminate = False
    nf = 0
    fopt = float(fbase)
    xopt = np.asarray(xbase, dtype=float).copy()
    fnew = fopt

    for j in range(len(direction_indices)):
        if nf >= submaxfun:
            terminate = True
            break
        di = direction_indices[j]
        xnew = xbase + alpha * D[:, di]
        fnew = float(fun(xnew))
        nf += 1
        if fnew < fopt:
            xopt = xnew.copy()
            fopt = fnew
        if fnew <= -np.inf or nf >= submaxfun:
            terminate = True
            break
        if fnew < fbase:
            # Ordinary direction cycling: try the successful direction first
            # next time this coordinate block is visited.
            reorder = [direction_indices[j]] + direction_indices[:j]
            direction_indices[: j + 1] = reorder
            break

    if fnew <= -np.inf:
        exitflag = 0
    elif nf >= submaxfun:
        exitflag = 1

    output = {"nf": nf, "direction_indices": direction_indices, "terminate": terminate}
    return xopt, fopt, exitflag, output


def _try_extrapolation(fun, xbase, fbase, direction, step, nf, maxfun):
    """Try up to two extrapolation steps along a remembered direction."""

    xbest = xbase
    fbest = fbase
    for _ in range(2):
        if nf >= maxfun:
            break
        xcand = xbest + step * direction
        fcand = float(fun(xcand))
        nf += 1
        if fcand < fbest:
            xbest = xcand.copy()
            fbest = fcand
            step *= 2.0
        else:
            break
    return xbest, fbest, nf


def _diagonal_directions(n):
    """Return a small hand-crafted diagonal direction set for low dimensions."""

    if not 2 <= n <= 10:
        return None

    d_all = np.ones(n, dtype=float) / np.sqrt(n)
    pairs = [d_all, -d_all]

    if n >= 3:
        d_alt = np.ones(n, dtype=float)
        d_alt[1::2] = -1.0
        d_alt /= np.linalg.norm(d_alt)
        if np.abs(np.dot(d_all, d_alt)) < 0.95:
            pairs.extend([d_alt, -d_alt])

    if n >= 4:
        d_dec = np.arange(n, 0, -1, dtype=float)
        d_dec /= np.linalg.norm(d_dec)
        if not any(np.abs(np.dot(d_dec, p)) > 0.95 for p in pairs):
            pairs.extend([d_dec, -d_dec])

    return np.column_stack(pairs)


def _remember_direction(prod_memory, direction, step, mem_size):
    """Store a productive direction unless a nearly parallel one exists."""

    direction = np.asarray(direction, dtype=float)
    norm_direction = np.linalg.norm(direction)
    if norm_direction == 0:
        return
    direction = direction / norm_direction
    is_dup = any(np.abs(np.dot(m[0], direction)) > 0.95 for m in prod_memory)
    if is_dup:
        return
    if len(prod_memory) >= mem_size:
        prod_memory.pop()
    prod_memory.insert(0, (direction.copy(), float(step)))


def _best_recorded_point(fopt_all, xopt_all, fopt, xopt):
    if np.any(~np.isnan(fopt_all)):
        idx = int(np.nanargmin(fopt_all))
        if fopt_all[idx] < fopt:
            return float(fopt_all[idx]), xopt_all[:, idx].copy()
    return fopt, xopt


def solver(fun, x0):
    x0 = np.asarray(x0, dtype=float).ravel()
    n = int(x0.size)
    maxfun = MAX_EVAL_FACTOR * n
    maxit = maxfun
    alpha_tol = 1e-6
    alpha_all = np.ones(n, dtype=float)
    expand = 2.0
    shrink = 0.5
    eps_m = np.finfo(float).eps

    # Coordinate directions: [+e1, -e1, +e2, -e2, ...]
    D = np.zeros((n, 2 * n), dtype=float)
    D[:, 0::2] = np.eye(n)
    D[:, 1::2] = -np.eye(n)
    grouped_direction_indices = [[2 * k, 2 * k + 1] for k in range(n)]

    diag_dirs = _diagonal_directions(n)
    mem_size = max(1, min(n, 5))
    prod_memory: list[tuple[np.ndarray, float]] = []
    momentum = np.zeros(n, dtype=float)
    momentum_decay = 0.6

    fopt_all = np.full(n, np.nan, dtype=float)
    xopt_all = np.full((n, n), np.nan, dtype=float)

    f0 = float(fun(x0))
    nf = 1
    xbase = x0.copy()
    fbase = f0
    xopt = x0.copy()
    fopt = f0
    stagnation_count = 0
    terminate = False

    for _iter in range(maxit):
        xbase_sweep_start = xbase.copy()
        fbase_sweep_start = fbase
        sweep_improved = False

        # Explicit productive displacement memory beyond cycling.
        if prod_memory and nf < maxfun:
            avg_alpha = float(np.mean(alpha_all))
            for m_idx in range(len(prod_memory)):
                if nf >= maxfun:
                    break
                dir_vec, stored_step = prod_memory[m_idx]
                step = max(avg_alpha, stored_step)
                x_cand = xbase + step * dir_vec
                f_cand = float(fun(x_cand))
                nf += 1
                if f_cand < fbase:
                    xbase = x_cand.copy()
                    fbase = f_cand
                    sweep_improved = True
                    xbase, fbase, nf = _try_extrapolation(
                        fun, xbase, fbase, dir_vec, step * 2.0, nf, maxfun
                    )
                    prod_memory.pop(m_idx)
                    prod_memory.insert(0, (dir_vec, step))
                    break

        # Baseline coordinate order; ordinary direction cycling is kept within
        # each coordinate block via grouped_direction_indices.
        for i in range(n):
            if terminate or nf >= maxfun:
                break
            direction_indices = list(grouped_direction_indices[i])
            sub_xopt, sub_fopt, _, sub_output = inner_direct_search(
                fun,
                xbase,
                fbase,
                D,
                direction_indices,
                float(alpha_all[i]),
                max(0, maxfun - nf),
            )
            nf += sub_output["nf"]
            fopt_all[i] = sub_fopt
            xopt_all[:, i] = sub_xopt
            grouped_direction_indices[i] = sub_output["direction_indices"]

            xold = xbase.copy()
            is_expand = sub_fopt + eps_m * (alpha_all[i] ** 2) < fbase
            alpha_all[i] *= expand if is_expand else shrink

            if sub_output["terminate"]:
                terminate = True
                break
            if sub_fopt < fbase:
                xbase = sub_xopt.copy()
                fbase = sub_fopt
                displacement = sub_xopt - xold
                disp_norm = np.linalg.norm(displacement)
                if disp_norm > alpha_tol:
                    _remember_direction(prod_memory, displacement, disp_norm, mem_size)
            if np.all(alpha_all < alpha_tol) or nf >= maxfun:
                terminate = True
                break

        displacement = xbase - xbase_sweep_start
        disp_norm = np.linalg.norm(displacement)
        if fbase < fbase_sweep_start:
            sweep_improved = True
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Sweep-level pattern / momentum extrapolation.
        if not terminate and sweep_improved and disp_norm > alpha_tol and nf < maxfun:
            pattern_dir = displacement / disp_norm
            alpha_pat = max(disp_norm, alpha_tol)

            momentum = momentum_decay * momentum + (1.0 - momentum_decay) * pattern_dir
            momentum_norm = np.linalg.norm(momentum)
            momentum_dir = momentum / momentum_norm if momentum_norm > alpha_tol else None

            factors = (1.0, 2.0, 4.0)
            x_pat = xbase.copy()
            f_pat = fbase
            best_dir = None
            pat_improved = False

            for factor in factors:
                if nf >= maxfun:
                    break
                x_candidate = xbase + factor * alpha_pat * pattern_dir
                f_candidate = float(fun(x_candidate))
                nf += 1
                if f_candidate < f_pat:
                    x_pat = x_candidate.copy()
                    f_pat = f_candidate
                    best_dir = pattern_dir
                    pat_improved = True
                else:
                    break

            if not pat_improved and momentum_dir is not None and nf < maxfun:
                for factor in factors:
                    if nf >= maxfun:
                        break
                    x_candidate = xbase + factor * alpha_pat * momentum_dir
                    f_candidate = float(fun(x_candidate))
                    nf += 1
                    if f_candidate < f_pat:
                        x_pat = x_candidate.copy()
                        f_pat = f_candidate
                        best_dir = momentum_dir
                    else:
                        break

            if f_pat < fbase:
                xbase = x_pat.copy()
                fbase = f_pat
                if best_dir is not None:
                    _remember_direction(prod_memory, best_dir, alpha_pat, mem_size)

        # Diagonal probing on stagnation, low dimensions only.
        if (
            not terminate
            and not sweep_improved
            and diag_dirs is not None
            and stagnation_count >= 1
            and nf < maxfun - diag_dirs.shape[1] + 1
        ):
            n_diag = diag_dirs.shape[1]
            diag_step = float(np.median(alpha_all))
            offset = (stagnation_count - 1) % max(1, n_diag // 2)
            for j in range(n_diag):
                if nf >= maxfun:
                    break
                idx = (j + offset) % n_diag
                x_cand = xbase + diag_step * diag_dirs[:, idx]
                f_cand = float(fun(x_cand))
                nf += 1
                if f_cand < fbase:
                    xbase = x_cand.copy()
                    fbase = f_cand
                    sweep_improved = True
                    stagnation_count = 0
                    _remember_direction(prod_memory, diag_dirs[:, idx], diag_step, mem_size)
                    break

        fopt, xopt = _best_recorded_point(fopt_all, xopt_all, fopt, xopt)
        if fbase < fopt:
            fopt = fbase
            xopt = xbase.copy()

        if nf >= maxfun or np.all(alpha_all < alpha_tol):
            terminate = True
        if terminate:
            break

    return xopt
