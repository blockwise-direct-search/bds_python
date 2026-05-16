# bds

Python implementation of the Blockwise Direct Search (BDS) method.

BDS solves unconstrained optimization problems without using derivatives. It
does not support bound constraints, linear constraints, nonlinear constraints,
or general constrained optimization.

The public API follows the conventions of `scipy.optimize`: the solver accepts
SciPy-style controls such as `maxiter`, `maxfev`, `xatol`, `fatol`, `tol`,
`disp`, and `return_all`, and returns an `OptimizeResult`-like object with
fields such as `x`, `fun`, `success`, `status`, `message`, `nfev`, `nit`, and
optional histories.

```python
import numpy as np
from bds import minimize_bds


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


res = minimize_bds(rosenbrock, np.array([-1.2, 1.0]))
print(res.x, res.fun, res.message)
```

The shorter alias `bds(...)` is also available.

## Development

Run the test suite from the repository root with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

SciPy is optional. When SciPy is installed, BDS returns SciPy's
`scipy.optimize.OptimizeResult`; otherwise it returns a compatible fallback
object.

## SciPy Compatibility

`minimize_bds` can also be used as a SciPy custom minimizer method. Derivative
keywords such as `jac`, `hess`, and `hessp` may be passed by
`scipy.optimize.minimize`; BDS ignores them with an `OptimizeWarning` because it
is derivative-free. Nonempty `bounds` or `constraints` are rejected, since BDS
is only for unconstrained problems.

Callbacks support both SciPy callback styles:

- `callback(xk)`
- `callback(intermediate_result=OptimizeResult(...))`

Raising `StopIteration` from a callback terminates the solver.
