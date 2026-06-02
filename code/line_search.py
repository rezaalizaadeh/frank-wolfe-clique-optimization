"""
line_search.py

Shared exact line-search utilities for the Frank-Wolfe variants.

For the project objective:

    F(x) = x^T A x + 0.5 ||x||^2

we solve the minimization version:

    f(x) = -F(x)

All three algorithms will update:

    x_new = x + gamma * d

where gamma is chosen in [0, gamma_max].
"""

import numpy as np

from objective import minimization_objective


def exact_line_search(A, x, d, gamma_max):
    """
    Exact line search for the minimization objective:

        f(x) = -x^T A x - 0.5 ||x||^2

    along the direction:

        x_new = x + gamma d

    with:

        gamma in [0, gamma_max]

    Since the objective is quadratic, f(x + gamma d) is a quadratic function
    in gamma, so we can compute the best gamma analytically.

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix.
    x : np.ndarray
        Current iterate.
    d : np.ndarray
        Search direction.
    gamma_max : float
        Maximum feasible step size.

    Returns
    -------
    float
        Chosen step size gamma.
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    if gamma_max < 0:
        raise ValueError("gamma_max must be nonnegative.")

    if gamma_max == 0:
        return 0.0

    # f(x + gamma d) = c + b gamma + a gamma^2
    # for f(x) = -x^T A x - 0.5 x^T x
    a = -(d @ A @ d) - 0.5 * (d @ d)
    b = -2.0 * (x @ A @ d) - (x @ d)

    # If the quadratic coefficient is almost zero, choose the best endpoint.
    if abs(a) < 1e-14:
        f0 = minimization_objective(A, x)
        f1 = minimization_objective(A, x + gamma_max * d)
        return 0.0 if f0 <= f1 else float(gamma_max)

    # Stationary point of c + b gamma + a gamma^2:
    # derivative = b + 2a gamma = 0
    gamma_star = -b / (2.0 * a)

    # Candidate points: left endpoint, right endpoint, and stationary point if feasible.
    candidates = [0.0, float(gamma_max)]

    if 0.0 <= gamma_star <= gamma_max:
        candidates.append(float(gamma_star))

    # Choose the gamma giving the smallest minimization objective.
    best_gamma = candidates[0]
    best_value = minimization_objective(A, x + best_gamma * d)

    for gamma in candidates[1:]:
        value = minimization_objective(A, x + gamma * d)
        if value < best_value:
            best_value = value
            best_gamma = gamma

    return float(best_gamma)


def clipped_step_size(gamma, gamma_max):
    """
    Clip a step size to the feasible interval [0, gamma_max].

    This is useful as a numerical safety function.

    Parameters
    ----------
    gamma : float
        Candidate step size.
    gamma_max : float
        Maximum feasible step size.

    Returns
    -------
    float
        Clipped step size.
    """
    return float(min(max(gamma, 0.0), gamma_max))