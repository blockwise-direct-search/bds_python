"""Exit status helpers."""

from __future__ import annotations

SMALL_ALPHA = 0
FTARGET_REACHED = 1
MAXFUN_REACHED = 2
MAXIT_REACHED = 3
SMALL_OBJECTIVE_CHANGE = 4
SMALL_ESTIMATED_GRADIENT = 5
GRADIENT_ESTIMATION_COMPLETED = 6
CALLBACK_STOP = 7


MESSAGES = {
    SMALL_ALPHA: "The step-size tolerance was reached.",
    FTARGET_REACHED: "The target objective value was reached.",
    MAXFUN_REACHED: "The maximum number of function evaluations was reached.",
    MAXIT_REACHED: "The maximum number of iterations was reached.",
    SMALL_OBJECTIVE_CHANGE: "The change in objective value is small.",
    SMALL_ESTIMATED_GRADIENT: "The estimated gradient is small.",
    GRADIENT_ESTIMATION_COMPLETED: "The gradient estimation is completed.",
    CALLBACK_STOP: "The callback raised StopIteration.",
}


SUCCESS_STATUSES = {
    SMALL_ALPHA,
    FTARGET_REACHED,
    SMALL_OBJECTIVE_CHANGE,
    SMALL_ESTIMATED_GRADIENT,
    GRADIENT_ESTIMATION_COMPLETED,
}
