# API

## `minimize_bds(fun, x0, args=(), options=None, callback=None, **options)`

Minimize an unconstrained scalar objective using Blockwise Direct Search. BDS
does not solve bound-constrained, linearly constrained, nonlinearly constrained,
or generally constrained problems.

The function returns an `OptimizeResult` compatible object. Important fields:

- `x`: best point found.
- `fun`: objective value at `x`.
- `success`: whether the termination status is successful.
- `status`: integer termination code.
- `message`: human-readable termination message.
- `nfev`: number of objective evaluations.
- `nit`: number of BDS outer iterations.
- `fhist`: objective-value history.
- `allvecs`: list of evaluated points when `return_all=True` or
  `output_xhist=True`.

Accepted option names include BDS-specific names such as
`MaxFunctionEvaluations`, `StepTolerance`, and `Algorithm`, plus SciPy-style
aliases such as `maxiter`, `maxfev`, `xatol`, `fatol`, `ftol`, `tol`, `disp`,
`return_all`, and `direc`.

The callable accepts derivative pass-through keywords such as `jac`, `hess`,
and `hessp` for SciPy custom-minimizer compatibility. They are ignored because
BDS is derivative-free. Nonempty `bounds` and `constraints` inputs raise
`ValueError`.
