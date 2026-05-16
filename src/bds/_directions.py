"""Polling direction helpers."""

from __future__ import annotations

import warnings

import numpy as np

from ._options import BDSOptions


def cycling(array: np.ndarray, index: int, strategy: int) -> np.ndarray:
    """Cycle a 1-D array of direction indices.

    This is used only by opportunistic polling. If a direction gives sufficient
    decrease, the next visit to the same block can start from that successful
    direction or from its successor. ``index`` is zero-based in this Python
    implementation.

    Strategies
    ----------
    0
        No permutation.
    1
        Move the successful element to the front.
    2
        Move the successful element and all following elements to the front.
    3
        Move the elements after the successful element to the front.
    """

    array = np.ravel(np.asarray(array, dtype=int)).copy()
    if index < 0 or strategy == 0:
        return array
    if index >= array.size:
        raise IndexError("index is out of bounds for cycling.")
    if strategy == 1:
        return np.concatenate(([array[index]], array[:index], array[index + 1 :]))
    if strategy == 2:
        return np.concatenate((array[index:], array[:index]))
    if strategy == 3:
        return np.concatenate((array[index + 1 :], array[: index + 1]))
    raise ValueError("strategy must be in {0, 1, 2, 3}.")


def divide_direction_set(n: int, num_blocks: int, options: BDSOptions | None = None) -> list[np.ndarray]:
    """Split paired directions into blocks.

    Returned direction indices are zero-based indices into the complete
    direction matrix ``[d_0, -d_0, d_1, -d_1, ...]``.

    If no user grouping is supplied, dimensions are divided as evenly as
    possible: the first ``n % num_blocks`` blocks receive one extra dimension.
    A user-supplied ``grouped_direction_indices`` groups dimensions, not signed
    directions; each dimension contributes both ``d_i`` and ``-d_i`` to the same
    block.
    """

    if options is None or options.grouped_direction_indices is None:
        counts = np.full(num_blocks, n // num_blocks, dtype=int)
        counts[: n % num_blocks] += 1
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        dim_groups = [np.arange(start, start + count) for start, count in zip(starts, counts)]
    else:
        dim_groups = [np.asarray(group, dtype=int) - 1 for group in options.grouped_direction_indices]

    groups = []
    for dims in dim_groups:
        direction_indices = []
        for dim in dims:
            direction_indices.extend([2 * dim, 2 * dim + 1])
        groups.append(np.asarray(direction_indices, dtype=int))
    return groups


def get_direction_set(n: int, options: BDSOptions) -> np.ndarray:
    """Return the complete polling direction set ``[d_i, -d_i]``.

    By default the positive directions are the coordinate basis. If the user
    supplies ``direction_set``, its columns define the positive directions.
    Degenerate input is repaired before the signed pairs are formed: nonfinite
    entries are replaced by zero, very short directions are removed, nearly
    parallel directions are collapsed, and missing directions are supplemented
    so that the final positive directions form a basis of the full space.
    """

    direction_set = np.asarray(options.direction_set, dtype=float).copy()

    if not np.all(np.isfinite(direction_set)):
        warnings.warn(
            "Some directions contain NaN or Inf and were replaced with 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        direction_set[~np.isfinite(direction_set)] = 0.0

    direction_set = _repair_direction_basis(direction_set)

    directions = np.empty((n, 2 * n), dtype=float)
    directions[:, 0::2] = direction_set
    directions[:, 1::2] = -direction_set
    return directions


def direction_probability_matrix(
    n: int,
    batch_size: int,
    grouped_direction_indices: list[np.ndarray],
    available_block_indices: np.ndarray,
) -> np.ndarray:
    """Diagonal matrix of probabilities for selecting each positive direction.

    Directions in unavailable blocks get probability zero. Available blocks are
    sampled uniformly, and every positive direction belonging to an available
    block receives the corresponding block-selection probability. This matrix is
    used by the randomized gradient estimator.
    """

    probabilities = np.zeros(n, dtype=float)
    if available_block_indices.size == 0:
        raise RuntimeError("No blocks are available for selection.")
    block_probability = batch_size / available_block_indices.size

    for block_idx in available_block_indices:
        direction_indices = grouped_direction_indices[block_idx]
        positive_direction_indices = direction_indices[direction_indices % 2 == 0] // 2
        probabilities[positive_direction_indices] = block_probability

    return np.diag(probabilities)


def _repair_direction_basis(direction_set: np.ndarray) -> np.ndarray:
    """Make a possibly poor user direction set into a full basis.

    Preserve an independent subset of user directions in their original order,
    then supplement it with coordinate directions until a basis is obtained.
    """

    n = direction_set.shape[0]
    shortest_norm = 10.0 * np.sqrt(n) * np.finfo(float).eps
    norms = np.linalg.norm(direction_set, axis=0)
    # The sqrt(n) factor keeps the notion of "too short to be useful" scaled to
    # the problem dimension.
    keep = norms >= shortest_norm
    if not np.all(keep):
        warnings.warn(
            f"Directions shorter than {shortest_norm:g} were removed.",
            RuntimeWarning,
            stacklevel=2,
        )
        direction_set = direction_set[:, keep]
        norms = norms[keep]

    if direction_set.size == 0 or direction_set.shape[1] == 0:
        return np.eye(n)

    # Preserve the first direction in each nearly parallel group and discard
    # later directions, so a user's earlier directions take precedence.
    keep_cols = []
    for j in range(direction_set.shape[1]):
        col = direction_set[:, j]
        col_norm = np.linalg.norm(col)
        parallel = False
        for kept in keep_cols:
            kept_col = direction_set[:, kept]
            dot = abs(np.dot(col, kept_col))
            if dot > (1.0 - 1e-10) * col_norm * np.linalg.norm(kept_col):
                parallel = True
                break
        if not parallel:
            keep_cols.append(j)
    if len(keep_cols) != direction_set.shape[1]:
        warnings.warn(
            "Nearly parallel directions were removed.",
            RuntimeWarning,
            stacklevel=2,
        )
    direction_set = direction_set[:, keep_cols]

    rank = np.linalg.matrix_rank(direction_set, tol=1e-10)
    if direction_set.shape == (n, n) and rank == n:
        return direction_set

    # Build a maximal linearly independent subset and supplement it with
    # coordinate directions if the user-provided directions do not span R^n.
    basis_cols = []
    current = np.empty((n, 0))
    for j in range(direction_set.shape[1]):
        candidate = direction_set[:, [j]]
        trial = np.hstack((current, candidate))
        if np.linalg.matrix_rank(trial, tol=1e-10) > current.shape[1]:
            basis_cols.append(direction_set[:, j])
            current = trial
        if len(basis_cols) == n:
            break

    for j in range(n):
        candidate = np.eye(n)[:, [j]]
        trial = np.hstack((current, candidate))
        if np.linalg.matrix_rank(trial, tol=1e-10) > current.shape[1]:
            basis_cols.append(candidate[:, 0])
            current = trial
        if len(basis_cols) == n:
            break

    repaired = np.column_stack(basis_cols)
    if repaired.shape != (n, n) or np.linalg.matrix_rank(repaired, tol=1e-10) < n:
        repaired = np.eye(n)
    warnings.warn(
        "direction_set was repaired to obtain a full basis.",
        RuntimeWarning,
        stacklevel=2,
    )
    return repaired
