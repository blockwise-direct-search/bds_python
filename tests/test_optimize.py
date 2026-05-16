import math
import unittest
import warnings

import numpy as np

from bds import bds, minimize_bds
from bds._exit import CALLBACK_STOP, FTARGET_REACHED, MAXFUN_REACHED, MAXIT_REACHED, SMALL_ALPHA
from bds._utils import eval_fun


class OptimizeTests(unittest.TestCase):
    def test_eval_fun_converts_nan_and_exceptions_to_infinity(self):
        nan_eval = eval_fun(lambda x: np.nan, np.zeros(2))
        self.assertTrue(math.isinf(nan_eval.value))
        self.assertTrue(math.isnan(nan_eval.raw_value))
        self.assertFalse(nan_eval.is_valid)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bad_eval = eval_fun(lambda x: (_ for _ in ()).throw(RuntimeError("boom")), np.zeros(2))
        self.assertTrue(any(item.category is RuntimeWarning for item in caught))
        self.assertTrue(math.isinf(bad_eval.value))
        self.assertTrue(math.isnan(bad_eval.raw_value))
        self.assertFalse(bad_eval.is_valid)

    def test_minimize_bds_reduces_convex_quadratic(self):
        target = np.array([1.0, -2.0, 0.5])

        def quadratic(x):
            return np.sum((x - target) ** 2)

        res = minimize_bds(
            quadratic,
            np.zeros(3),
            options={
                "Algorithm": "cbds",
                "MaxFunctionEvaluations": 2000,
                "StepTolerance": 1e-7,
                "seed": 1,
                "output_xhist": True,
                "output_alpha_hist": True,
                "output_block_hist": True,
            },
        )

        self.assertTrue(res.success)
        self.assertEqual(res.status, SMALL_ALPHA)
        self.assertLess(res.fun, 1e-10)
        np.testing.assert_allclose(res.x, target, atol=1e-4)
        self.assertEqual(res.nfev, res.fhist.size)
        self.assertEqual(res.xhist.shape[0], 3)
        self.assertEqual(res.alpha_hist.shape[0], 3)
        self.assertGreater(res.blocks_hist.size, 0)

    def test_alias_bds_and_scipy_like_options_work(self):
        res = bds(
            lambda x: (x[0] - 2.0) ** 2,
            [10.0],
            maxfev=500,
            xatol=1e-8,
            seed=42,
        )

        self.assertTrue(res.success)
        np.testing.assert_allclose(res.x, [2.0], atol=1e-4)

    def test_ftarget_and_maxfev_statuses(self):
        res_target = minimize_bds(
            lambda x: np.sum(x**2),
            [0.0, 0.0],
            options={"ftarget": 0.0},
        )
        self.assertEqual(res_target.status, FTARGET_REACHED)
        self.assertEqual(res_target.nfev, 1)

        res_budget = minimize_bds(
            lambda x: np.sum(x**2),
            [1.0, 1.0],
            options={"MaxFunctionEvaluations": 1},
        )
        self.assertEqual(res_budget.status, MAXFUN_REACHED)
        self.assertEqual(res_budget.nfev, 1)

    def test_rosenbrock_smoke(self):
        def rosenbrock(x):
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

        res = minimize_bds(
            rosenbrock,
            np.array([-1.2, 1.0]),
            options={
                "Algorithm": "cbds",
                "MaxFunctionEvaluations": 10000,
                "StepTolerance": 1e-5,
                "alpha_init": "auto",
                "seed": 0,
            },
        )

        self.assertTrue(res.success)
        self.assertLess(res.fun, 1e-5)
        np.testing.assert_allclose(res.x, [1.0, 1.0], atol=5e-3)

    def test_scipy_custom_method_kwargs_are_accepted(self):
        res = minimize_bds(
            lambda x: np.sum((x - 1.0) ** 2),
            [3.0, 3.0],
            jac=lambda x: 2 * (x - 1.0),
            hess=lambda x: np.eye(2),
            bounds=None,
            constraints=(),
            tol=1e-6,
            maxiter=20,
            maxfev=200,
            return_all=True,
            direc=np.eye(2),
            seed=0,
        )

        self.assertIn(res.status, {SMALL_ALPHA, MAXIT_REACHED, MAXFUN_REACHED})
        self.assertTrue(hasattr(res, "allvecs"))
        self.assertGreaterEqual(len(res.allvecs), 1)

    def test_bounds_and_constraints_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support bounds"):
            minimize_bds(
                lambda x: np.sum(x**2),
                [1.0, 1.0],
                bounds=[(0.0, None), (None, None)],
            )

        with self.assertRaisesRegex(ValueError, "does not support constraints"):
            minimize_bds(
                lambda x: np.sum(x**2),
                [1.0, 1.0],
                constraints=({"type": "ineq", "fun": lambda x: x[0]},),
            )

    def test_callback_accepts_xk_and_stop_iteration(self):
        calls = []

        def callback(xk):
            calls.append(xk.copy())
            raise StopIteration

        res = minimize_bds(
            lambda x: np.sum(x**2),
            [1.0, 1.0],
            callback=callback,
            options={"maxfev": 100},
        )

        self.assertEqual(res.status, CALLBACK_STOP)
        self.assertEqual(len(calls), 1)

    def test_callback_accepts_intermediate_result_keyword(self):
        values = []

        def callback(intermediate_result):
            values.append(intermediate_result.fun)
            raise StopIteration

        res = minimize_bds(
            lambda x: np.sum(x**2),
            [1.0, 1.0],
            callback=callback,
            options={"maxfev": 100},
        )

        self.assertEqual(res.status, CALLBACK_STOP)
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], float)


if __name__ == "__main__":
    unittest.main()
