"""BDS-style grouped coordinate direct search (one-page MATLAB reference).

OptiProfiler expects ``solver(fun, x0)``. NumPy only.
"""
from __future__ import annotations

import numpy as np


def inner_direct_search(fun, xbase, fbase, D, direction_indices, alpha, submaxfun):
    direction_indices = list(direction_indices)
    exitflag = np.nan
    terminate = False
    nf = 0
    fopt = float(fbase)
    xopt = np.asarray(xbase, dtype=float).copy()
    fnew = fopt

    for j in range(len(direction_indices)):
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
            reorder = [direction_indices[j]] + direction_indices[:j]
            direction_indices[: j + 1] = reorder
            break

    if fnew <= -np.inf:
        exitflag = 0
    elif nf >= submaxfun:
        exitflag = 1

    output = {"nf": nf, "direction_indices": direction_indices, "terminate": terminate}
    return xopt, fopt, exitflag, output


def solver(fun, x0):
    x0 = np.asarray(x0, dtype=float).ravel()
    n = int(x0.size)
    maxfun = 500 * n
    maxit = maxfun
    alpha_tol = 1e-6
    alpha_all = np.ones(n, dtype=float)
    expand = 2.0
    shrink = 0.5
    eps_m = np.finfo(float).eps

    D = np.zeros((n, 2 * n), dtype=float)
    D[:, 0::2] = np.eye(n)
    D[:, 1::2] = -np.eye(n)

    grouped_direction_indices = [[2 * k, 2 * k + 1] for k in range(n)]

    fopt_all = np.full(n, np.nan, dtype=float)
    xopt_all = np.full((n, n), np.nan, dtype=float)
    terminate = False

    f0 = float(fun(x0))
    nf = 1
    xbase = x0.copy()
    fbase = f0
    xopt = x0.copy()
    fopt = f0

    for _iter in range(maxit):
        for i in range(n):
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

            is_expand = sub_fopt + eps_m * (alpha_all[i] ** 2) < fbase
            alpha_all[i] *= expand if is_expand else shrink

            if sub_output["terminate"]:
                terminate = True
                break
            if sub_fopt < fbase:
                xbase = sub_xopt.copy()
                fbase = sub_fopt
            if np.all(alpha_all < alpha_tol):
                terminate = True
                break
            if nf >= maxfun:
                terminate = True
                break

        if np.any(~np.isnan(fopt_all)):
            idx = int(np.nanargmin(fopt_all))
            if fopt_all[idx] < fopt:
                fopt = float(fopt_all[idx])
                xopt = xopt_all[:, idx].copy()

        if terminate:
            break

    return xopt
