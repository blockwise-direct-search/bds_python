import unittest

import numpy as np

from bds._gradient import GradientInfo, estimate_gradient, gradient_error_bound


class GradientEstimationTests(unittest.TestCase):
    def test_estimate_gradient_from_complete_coordinate_poll(self):
        n = 4
        x = np.array([1.5, -0.5, 0.25, 2.0])
        center = np.array([0.5, -1.0, 1.25, -0.5])
        alpha = 1e-3

        complete_direction_set = np.empty((n, 2 * n))
        complete_direction_set[:, 0::2] = np.eye(n)
        complete_direction_set[:, 1::2] = -np.eye(n)
        direction_indices = np.arange(2 * n)

        def quadratic(z):
            return float(np.sum((z - center) ** 2))

        values = np.array(
            [quadratic(x + alpha * complete_direction_set[:, j]) for j in direction_indices]
        )
        info = GradientInfo(
            n=n,
            complete_direction_set=complete_direction_set,
            step_size_per_batch=np.array([alpha]),
            step_size_per_block=np.full(n, alpha),
            fbase_per_batch=np.array([quadratic(x)]),
            direction_selection_probability_matrix=np.eye(n),
            sampled_direction_indices_per_batch=[direction_indices],
            function_values_per_batch=[values],
        )

        grad = estimate_gradient(info)

        np.testing.assert_allclose(grad, 2.0 * (x - center), rtol=1e-10, atol=1e-10)

    def test_gradient_error_bound_is_finite_and_nonnegative(self):
        n = 3
        alpha = np.array([1e-2, 2e-2, 3e-2])
        positive_direction_set = np.eye(n)
        grouped_direction_indices = [np.array([2 * i, 2 * i + 1]) for i in range(n)]

        bound = gradient_error_bound(
            alpha,
            batch_size=n,
            direction_indices_per_block=grouped_direction_indices,
            n=n,
            positive_direction_set=positive_direction_set,
            direction_selection_probability_matrix=np.eye(n),
            lipschitz_constant=10.0,
        )

        self.assertTrue(np.isfinite(bound))
        self.assertGreaterEqual(bound, 0.0)


if __name__ == "__main__":
    unittest.main()
