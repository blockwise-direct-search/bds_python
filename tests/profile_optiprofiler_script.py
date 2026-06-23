#!/usr/bin/env python3
"""Default OptiProfiler comparison for the two local competitor solvers."""

from __future__ import annotations

from profile_optiprofiler import profile_optiprofiler


def main() -> None:
    options = {
        "mindim": 2,
        "maxdim": 50,
        "plibs": "s2mpj",
        "feature_name": "plain",
        "max_eval_factor": 200,
        "solver_names": ["bds", "evolved-bds"],
    }
    profile_optiprofiler(options)


if __name__ == "__main__":
    main()
