"""BDS with sign preference, diagonal probing, and pattern-validation re-poll.

Extends the grouped coordinate direct search with:
- Per-coordinate success-weighted sign ordering (try the winning sign first)
- Diagonal probing on stagnation to escape non-separable valleys
- Hooke-Jeeves style pattern re-poll after successful extrapolation (low dims)
- Dimension-aware activation of extra machinery

OptiProfiler expects ``solver(fun, x0)``. NumPy only.
"""
from __future__ import annotations

import numpy as np


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


def _try_extrapolation(fun, xbase, fbase, direction, step, nf, maxfun):
    """Try up to 2 extrapolation steps along *direction* starting from *xbase*.

    Each successful step doubles the step size for the next attempt.
    Returns ``(xbest, fbest, nf)`` with the best point found.
    """
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

    # Coordinate directions: [+e1, -e1, +e2, -e2, ...]
    D = np.zeros((n, 2 * n), dtype=float)
    D[:, 0::2] = np.eye(n)
    D[:, 1::2] = -np.eye(n)

    grouped_direction_indices = [[2 * k, 2 * k + 1] for k in range(n)]

    # --- Per-coordinate sign preference ---
    # Tracks which of {+e_i, -e_i} has been more successful recently.
    # > 0  -> try +e_i first   ;  < 0  -> try -e_i first
    sign_pref = np.ones(n, dtype=float)
    sign_pref_rate = 0.3  # EMA adaptation rate for the preference signal

    # --- Diagonal direction set for stagnation probing ---
    # Built once; only used when axis-aligned sweeps fail to improve.
    diag_dirs = None
    if 2 <= n <= 10:
        # Normalised all-ones diagonal
        d_all = np.ones(n, dtype=float) / np.sqrt(n)
        pairs = [d_all, -d_all]
        if n >= 3:
            # Alternating-sign diagonal  [1, -1, 1, -1, ...]
            d_alt = np.ones(n, dtype=float)
            d_alt[1::2] = -1.0
            d_alt /= np.linalg.norm(d_alt)
            # Only add if sufficiently different from d_all
            if np.abs(np.dot(d_all, d_alt)) < 0.95:
                pairs.extend([d_alt, -d_alt])
        if n >= 4:
            # Linearly decreasing weights  [n, n-1, ..., 1], normalised
            d_dec = np.arange(n, 0, -1, dtype=float)
            d_dec /= np.linalg.norm(d_dec)
            is_dup = any(np.abs(np.dot(d_dec, p)) > 0.95 for p in pairs)
            if not is_dup:
                pairs.extend([d_dec, -d_dec])
        diag_dirs = np.column_stack(pairs)  # (n, n_diag)

    # Exponential moving average success rate per coordinate (0.0 – 1.0).
    success_rate = np.full(n, 0.5, dtype=float)
    success_ema = 0.7

    # Track consecutive failures for step-size recovery.
    consecutive_failures = np.zeros(n, dtype=int)
    recovery_threshold = 5
    recovery_expand = 10.0

    # --- Productive direction memory ---
    mem_size = max(1, min(n, 5))
    prod_memory: list[tuple[np.ndarray, float]] = []

    # --- Momentum vector for pattern moves ---
    momentum = np.zeros(n, dtype=float)
    momentum_decay = 0.6
    momentum_active = False

    fopt_all = np.full(n, np.nan, dtype=float)
    xopt_all = np.full((n, n), np.nan, dtype=float)
    terminate = False

    f0 = float(fun(x0))
    nf = 1
    xbase = x0.copy()
    fbase = f0
    xopt = x0.copy()
    fopt = f0

    stagnation_count = 0

    for _iter in range(maxit):
        xbase_sweep_start = xbase.copy()
        fbase_sweep_start = fbase
        sweep_improved = False

        # --- 1. Opportunistic polling with productive direction memory ---
        if prod_memory and nf < maxfun:
            avg_alpha = float(np.mean(alpha_all))
            for m_idx in range(len(prod_memory)):
                if nf >= maxfun:
                    break
                dir_vec, _ = prod_memory[m_idx]

                step = avg_alpha
                x_cand = xbase + step * dir_vec
                f_cand = float(fun(x_cand))
                nf += 1
                if f_cand < fbase:
                    xbase = x_cand.copy()
                    fbase = f_cand
                    sweep_improved = True

                    xbase, fbase, nf = _try_extrapolation(
                        fun, xbase, fbase, dir_vec, step * 2.0, nf, maxfun,
                    )

                    prod_memory.pop(m_idx)
                    prod_memory.insert(0, (dir_vec, step))
                    break

                if n <= 5 and nf < maxfun:
                    x_cand2 = xbase - step * dir_vec
                    f_cand2 = float(fun(x_cand2))
                    nf += 1
                    if f_cand2 < fbase:
                        xbase = x_cand2.copy()
                        fbase = f_cand2
                        sweep_improved = True

                        xbase, fbase, nf = _try_extrapolation(
                            fun, xbase, fbase, -dir_vec, step * 2.0,
                            nf, maxfun,
                        )

                        prod_memory.pop(m_idx)
                        prod_memory.insert(0, (dir_vec, step))
                        break

        # --- 2. Adaptive coordinate ordering (highest success rate first) ---
        coord_order = np.argsort(-success_rate)

        # --- 3. Coordinate polling sweep ---
        for idx in range(n):
            if terminate or nf >= maxfun:
                break
            i = int(coord_order[idx])

            # Apply sign preference to order the direction pair
            if sign_pref[i] >= 0:
                direction_indices = [2 * i, 2 * i + 1]
            else:
                direction_indices = [2 * i + 1, 2 * i]

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
            was_success = sub_fopt < fbase
            is_expand = sub_fopt + eps_m * (alpha_all[i] ** 2) < fbase

            # Update step size
            if is_expand:
                alpha_all[i] *= expand
            else:
                alpha_all[i] *= shrink
            alpha_all[i] = max(alpha_all[i], alpha_tol)

            # Update sign preference based on which direction succeeded
            if was_success:
                # sub_output["direction_indices"][0] = the winning direction
                winner = sub_output["direction_indices"][0]
                if winner % 2 == 0:  # even = +e_i
                    sign_pref[i] = ((1.0 - sign_pref_rate) * sign_pref[i]
                                    + sign_pref_rate * 1.0)
                else:                 # odd  = -e_i
                    sign_pref[i] = ((1.0 - sign_pref_rate) * sign_pref[i]
                                    + sign_pref_rate * (-1.0))
                sign_pref[i] = max(-1.0, min(1.0, sign_pref[i]))

                success_rate[i] = (success_ema * success_rate[i]
                                   + (1.0 - success_ema) * 1.0)
                consecutive_failures[i] = 0
                xbase = sub_xopt.copy()
                fbase = sub_fopt

                # --- Add successful displacement to productive memory ---
                displacement = sub_xopt - xold
                disp_norm = np.linalg.norm(displacement)
                if disp_norm > alpha_tol:
                    dir_vec = displacement / disp_norm
                    is_dup = any(
                        np.abs(np.dot(m[0], dir_vec)) > 0.95
                        for m in prod_memory
                    )
                    if not is_dup:
                        if len(prod_memory) >= mem_size:
                            prod_memory.pop()
                        prod_memory.insert(0, (dir_vec, disp_norm))

                # --- Per-coordinate immediate pattern extension ---
                if nf < maxfun - 1:
                    x_ext = sub_xopt + displacement
                    f_ext = float(fun(x_ext))
                    nf += 1
                    if f_ext < fbase:
                        xbase = x_ext.copy()
                        fbase = f_ext
                        success_rate[i] = (success_ema * success_rate[i]
                                           + (1.0 - success_ema) * 1.0)

                        if nf < maxfun - 1:
                            x_ext2 = x_ext + displacement
                            f_ext2 = float(fun(x_ext2))
                            nf += 1
                            if f_ext2 < fbase:
                                xbase = x_ext2.copy()
                                fbase = f_ext2
            else:
                success_rate[i] = (success_ema * success_rate[i]
                                   + (1.0 - success_ema) * 0.0)
                consecutive_failures[i] += 1

                if (consecutive_failures[i] >= recovery_threshold
                        and alpha_all[i] < 1e-3
                        and nf < maxfun - n):
                    alpha_all[i] = min(1.0, recovery_expand * alpha_all[i])
                    consecutive_failures[i] = 0

            if sub_output["terminate"]:
                terminate = True
                break
            if nf >= maxfun:
                terminate = True
                break

        # --- 4. Check sweep improvement ---
        displacement = xbase - xbase_sweep_start
        disp_norm = np.linalg.norm(displacement)
        if fbase < fbase_sweep_start:
            sweep_improved = True
            stagnation_count = 0
        else:
            stagnation_count += 1

        # --- 5. Momentum-enhanced pattern / extrapolation move ---
        if sweep_improved and disp_norm > alpha_tol and nf < maxfun:
            pattern_dir = displacement / disp_norm
            alpha_pat = max(disp_norm, alpha_tol)

            momentum = (momentum_decay * momentum
                        + (1.0 - momentum_decay) * pattern_dir)
            momentum_norm = np.linalg.norm(momentum)
            if momentum_norm > alpha_tol:
                momentum_dir = momentum / momentum_norm
                momentum_active = True
            else:
                momentum_active = False

            factors = (1.0, 2.0, 4.0, 8.0) if n <= 5 else (1.0, 2.0, 4.0)

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

            if not pat_improved and momentum_active and nf < maxfun:
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

                # --- 5b. Pattern re-poll (Hooke-Jeeves validation) ---
                # After a successful pattern extrapolation, do a limited
                # coordinate re-poll at the pattern point.  If the pattern
                # point holds up under local exploration, we get extra
                # progress; if not, we already have the pattern benefit.
                # Active only for n <= 5 where the re-poll cost is small.
                if n <= 5 and n >= 2 and nf < maxfun - 2:
                    # Pick the coordinates with largest step sizes —
                    # these are most likely to be in a responsive region.
                    alpha_median = np.median(alpha_all)
                    active = np.where(alpha_all >= alpha_median)[0]
                    active = active[np.argsort(-success_rate[active])]
                    # Limit to 3 coordinates to contain cost
                    active = active[:min(3, len(active))]

                    pat_disp = np.zeros(n, dtype=float)
                    re_success = False
                    for ci in active:
                        if nf >= maxfun:
                            break
                        c = int(ci)
                        d1, d2 = (2 * c, 2 * c + 1) if sign_pref[c] >= 0 else (2 * c + 1, 2 * c)
                        for di in (d1, d2):
                            if nf >= maxfun:
                                break
                            xc = xbase + alpha_all[c] * D[:, di]
                            fc = float(fun(xc))
                            nf += 1
                            if fc < fbase:
                                pat_disp += alpha_all[c] * D[:, di]
                                xbase = xc.copy()
                                fbase = fc
                                re_success = True
                                break

                    # If the re-poll found improvement, try accelerating
                    # along the net displacement from the re-polls.
                    if re_success and nf < maxfun - 1:
                        pn = np.linalg.norm(pat_disp)
                        if pn > alpha_tol:
                            xc2 = xbase + pn * (pat_disp / pn)
                            fc2 = float(fun(xc2))
                            nf += 1
                            if fc2 < fbase:
                                xbase = xc2.copy()
                                fbase = fc2

                # Add to productive memory
                if best_dir is not None:
                    is_dup = any(
                        np.abs(np.dot(m[0], best_dir)) > 0.95
                        for m in prod_memory
                    )
                    if not is_dup:
                        if len(prod_memory) >= mem_size:
                            prod_memory.pop()
                        prod_memory.insert(0, (best_dir, alpha_pat))

        # --- 6. Diagonal probing on stagnation ---
        # When the coordinate sweep and pattern move both fail to improve,
        # try non-axis-aligned diagonal directions.  This helps on
        # rotated / non-separable problems where axis-aligned steps
        # stall.  Only active for 2 <= n <= 10.
        if (not sweep_improved and diag_dirs is not None
                and stagnation_count >= 1
                and nf < maxfun - diag_dirs.shape[1] + 1):
            n_diag = diag_dirs.shape[1]
            diag_step = float(np.median(alpha_all))
            # Alternate which diagonals we try on successive stagnation
            # events, so the probe diversity increases over time.
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

                    # Add successful diagonal to productive memory
                    dvec = diag_dirs[:, idx].copy()
                    is_dup = any(
                        np.abs(np.dot(m[0], dvec)) > 0.95
                        for m in prod_memory
                    )
                    if not is_dup:
                        if len(prod_memory) >= mem_size:
                            prod_memory.pop()
                        prod_memory.insert(0, (dvec, diag_step))
                    break

        # --- 7. Update overall best point ---
        if np.any(~np.isnan(fopt_all)):
            b_idx = int(np.nanargmin(fopt_all))
            if fopt_all[b_idx] < fopt:
                fopt = float(fopt_all[b_idx])
                xopt = xopt_all[:, b_idx].copy()

        # --- Termination checks ---
        if not terminate:
            if nf >= maxfun:
                terminate = True
            elif np.all(alpha_all < alpha_tol):
                terminate = True

        if terminate:
            break

    return xopt
