"""Gradient-estimation helpers used by BDS stopping criteria.

The estimate is used only when all selected blocks have been completely polled
and none achieved sufficient decrease. In that case the sampled positive and
negative direction values provide finite-difference information around the
current base point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GradientInfo:
    n: int
    complete_direction_set: np.ndarray
    step_size_per_batch: np.ndarray
    step_size_per_block: np.ndarray
    fbase_per_batch: np.ndarray
    direction_selection_probability_matrix: np.ndarray | None = None
    sampled_direction_indices_per_batch: list[np.ndarray] | None = None
    function_values_per_batch: list[np.ndarray] | None = None


def estimate_gradient(info: GradientInfo) -> np.ndarray:
    """Estimate the gradient from sampled paired direction values.

    The complete direction set stores signed pairs as
    ``[d_0, -d_0, d_1, -d_1, ...]``. If either signed direction from a dimension
    was sampled, the positive direction ``d_i`` is included in the basis for the
    estimate. Directional derivatives are central differences when both signed
    directions are available.
    """

    sampled_direction_indices = info.sampled_direction_indices_per_batch
    function_values = info.function_values_per_batch
    if sampled_direction_indices is None or function_values is None:
        raise ValueError("GradientInfo lacks sampled directions or function values.")

    # Concatenate the direction indices visited in this outer iteration, then
    # recover the dimensions represented by those signed directions.
    all_sampled = np.concatenate(sampled_direction_indices)
    sampled_dimensions = []
    for dim in range(info.n):
        if np.any(all_sampled == 2 * dim) or np.any(all_sampled == 2 * dim + 1):
            sampled_dimensions.append(dim)
    sampled_dimensions = np.asarray(sorted(sampled_dimensions), dtype=int)

    directional_derivatives = np.empty(sampled_dimensions.size, dtype=float)
    for j, dim in enumerate(sampled_dimensions):
        directional_derivatives[j] = _estimate_directional_derivative(
            dim,
            info.step_size_per_batch,
            function_values,
            sampled_direction_indices,
        )

    # Use a basic least-squares solve for the estimator equation. This avoids a
    # minimum-norm underdetermined solve, which would bias the randomized
    # gradient estimate toward smaller components.
    complete_basis = info.complete_direction_set[:, 0::2]
    sampled_basis = info.complete_direction_set[:, 2 * sampled_dimensions]
    probability = info.direction_selection_probability_matrix
    if probability is None:
        probability = np.eye(info.n)

    lhs = complete_basis @ probability @ complete_basis.T
    rhs = sampled_basis @ directional_derivatives
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def gradient_error_bound(
    alpha_all: np.ndarray,
    batch_size: int,
    direction_indices_per_block: list[np.ndarray],
    n: int,
    positive_direction_set: np.ndarray,
    direction_selection_probability_matrix: np.ndarray,
    lipschitz_constant: float,
) -> float:
    """Compute an upper bound on the gradient-estimation error.

    The bound uses all block step sizes, not only the latest sampled batch,
    because unvisited blocks still affect the uncertainty in the full gradient.
    ``lipschitz_constant`` is the user-provided Hessian/Lipschitz scale used to
    make the stopping test robust to finite-difference error.
    """

    alpha_full = np.zeros(n, dtype=float)
    num_blocks = len(direction_indices_per_block)
    for block_idx, indices in enumerate(direction_indices_per_block):
        positive_indices = indices // 2
        alpha_full[positive_indices] = alpha_all[block_idx]

    alpha_powers = alpha_full**4
    direction_norm_powers = np.linalg.norm(positive_direction_set, axis=0) ** 6
    scale = np.sqrt(np.sum(direction_norm_powers * alpha_powers))

    if batch_size == num_blocks:
        singular = _smallest_singular_value(positive_direction_set)
        return lipschitz_constant * scale / (6.0 * singular)

    weighted = positive_direction_set @ direction_selection_probability_matrix @ positive_direction_set.T
    weighted_singular = _smallest_singular_value(weighted)
    largest = np.linalg.svd(positive_direction_set, compute_uv=False)[0]
    return (
        lipschitz_constant
        * largest
        * (batch_size / num_blocks)
        * scale
        / (6.0 * weighted_singular)
    )


def _estimate_directional_derivative(
    dimension: int,
    step_size_per_batch: np.ndarray,
    function_values_per_batch: list[np.ndarray],
    sampled_direction_indices_per_batch: list[np.ndarray],
) -> float:
    """Estimate one directional derivative from a sampled signed pair."""

    positive_idx = 2 * dimension
    negative_idx = 2 * dimension + 1
    for batch_idx, direction_indices in enumerate(sampled_direction_indices_per_batch):
        positive_positions = np.flatnonzero(direction_indices == positive_idx)
        negative_positions = np.flatnonzero(direction_indices == negative_idx)
        if positive_positions.size and negative_positions.size:
            # Central difference. Forward/backward-only cases are deliberately
            # not used because the estimator relies on paired direction values.
            f_pos = function_values_per_batch[batch_idx][positive_positions[0]]
            f_neg = function_values_per_batch[batch_idx][negative_positions[0]]
            return (f_pos - f_neg) / (2.0 * step_size_per_batch[batch_idx])
    return np.nan


def _smallest_singular_value(matrix: np.ndarray) -> float:
    values = np.linalg.svd(matrix, compute_uv=False)
    smallest = values[-1]
    return max(float(smallest), np.finfo(float).tiny)
