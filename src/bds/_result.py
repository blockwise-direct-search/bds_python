"""Small compatibility layer for SciPy's ``OptimizeResult``."""

try:  # pragma: no cover - exercised only when SciPy is installed.
    from scipy.optimize import OptimizeResult as OptimizeResult
except Exception:  # pragma: no cover - fallback is covered in this environment.

    class OptimizeResult(dict):
        """Dictionary subclass exposing keys as attributes.

        This mirrors the public behavior relied on from
        ``scipy.optimize.OptimizeResult`` without making SciPy a hard runtime
        dependency.
        """

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
