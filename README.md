# bds

[![Unit test of BDS](https://github.com/blockwise-direct-search/bds_python/actions/workflows/unit_test.yml/badge.svg)](https://github.com/blockwise-direct-search/bds_python/actions/workflows/unit_test.yml)
[![Gradient test of BDS](https://github.com/blockwise-direct-search/bds_python/actions/workflows/gradient_test.yml/badge.svg)](https://github.com/blockwise-direct-search/bds_python/actions/workflows/gradient_test.yml)
[![Stress test of BDS](https://github.com/blockwise-direct-search/bds_python/actions/workflows/stress_test.yml/badge.svg)](https://github.com/blockwise-direct-search/bds_python/actions/workflows/stress_test.yml)
[![Parallel test of BDS](https://github.com/blockwise-direct-search/bds_python/actions/workflows/parallel_test.yml/badge.svg)](https://github.com/blockwise-direct-search/bds_python/actions/workflows/parallel_test.yml)
[![Recursive test of BDS](https://github.com/blockwise-direct-search/bds_python/actions/workflows/recursive_test.yml/badge.svg)](https://github.com/blockwise-direct-search/bds_python/actions/workflows/recursive_test.yml)

Python implementation of the Blockwise Direct Search (BDS) method.

BDS is designed for unconstrained derivative-free optimization.

The public API follows the conventions of `scipy.optimize`: the solver accepts
SciPy-style controls such as `maxiter`, `maxfev`, `xatol`, `fatol`, `tol`,
`disp`, and `return_all`, and returns an `OptimizeResult`-like object with
fields such as `x`, `fun`, `success`, `status`, `message`, `nfev`, `nit`, and
optional histories.

## Installation

For local development, install the package in editable mode from the repository
root:

```bash
python -m pip install -e .
```

## Quick Start

```python
import numpy as np
from bds import minimize_bds


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


res = minimize_bds(rosenbrock, np.array([-1.2, 1.0]))
print(res.x, res.fun, res.message)
```

The shorter alias `bds(...)` is also available.

SciPy is optional. When SciPy is installed, BDS returns SciPy's
`scipy.optimize.OptimizeResult`; otherwise it returns a compatible fallback
object.

## Testing

Run the full local test suite from the repository root with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## SciPy-Style Interface

`minimize_bds` returns an `OptimizeResult`-like object and accepts common
SciPy-style controls such as `maxiter`, `maxfev`, `xatol`, `fatol`, `tol`,
`disp`, and `return_all`. Arguments outside the BDS problem class, including
`jac`, `hess`, `hessp`, `bounds`, and `constraints`, are rejected with
`ValueError`.

Callbacks support both SciPy callback styles:

- `callback(xk)`
- `callback(intermediate_result=OptimizeResult(...))`

Raising `StopIteration` from a callback terminates the solver.
