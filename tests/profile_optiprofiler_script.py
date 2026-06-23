#!/usr/bin/env python3
"""Default OptiProfiler comparison for local competitor solvers."""

from __future__ import annotations

from profile_optiprofiler import profile_optiprofiler


def main() -> None:
    options = {
        "mindim": 6,
        "maxdim": 50,
        "plibs": "s2mpj",
        "feature_name": "linearly_transformed",
        "max_eval_factor": 200,
        "solver_names": ["evolved-bds-lean", "nomad"],
    }
    profile_optiprofiler(options)


if __name__ == "__main__":
    main()
