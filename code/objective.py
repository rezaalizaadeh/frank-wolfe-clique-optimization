"""
objective.py

Shared objective functions for the ODS26 Group 11 project.

Problem:
    L2-regularized maximum clique formulation.

Original maximization problem from the clustering paper:
    maximize F(x) = x^T A x + 0.5 ||x||_2^2
    subject to x in the simplex.

Frank-Wolfe algorithms are usually written for minimization, so we solve:
    minimize f(x) = -F(x)
    subject to x in the simplex.
"""

import numpy as np


def validate_adjacency_matrix(A, tol=1e-10):
    """
    Check whether A is a valid adjacency matrix for our experiments.

    Expected properties:
    - square matrix
    - symmetric
    - zero diagonal
    - nonnegative entries

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    tol : float
        Numerical tolerance.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If A does not satisfy the required properties.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if not np.allclose(A, A.T, atol=tol):
        raise ValueError("A must be symmetric.")

    if not np.allclose(np.diag(A), 0.0, atol=tol):
        raise ValueError("A must have zero diagonal.")

    if np.any(A < -tol):
        raise ValueError("A must have nonnegative entries.")


def validate_vector(A, x):
    """
    Check whether x has the correct shape for matrix A.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix with shape (n, n).
    x : np.ndarray
        Vector expected to have shape (n,).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If x has the wrong shape.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a 1D vector.")

    if A.shape[0] != x.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A has size {A.shape[0]}, but x has size {x.shape[0]}."
        )


def max_clique_objective(A, x, check_input=False):
    """
    Compute the maximization objective:

        F(x) = x^T A x + 0.5 ||x||_2^2

    This is the L2-regularized maximum clique objective.

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix of the graph.
    x : np.ndarray
        Current iterate.
    check_input : bool
        If True, validate A and x before computing the objective.

    Returns
    -------
    float
        Value of F(x).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)

    if check_input:
        validate_adjacency_matrix(A)
        validate_vector(A, x)

    return float(x @ A @ x + 0.5 * (x @ x))


def minimization_objective(A, x, check_input=False):
    """
    Compute the minimization objective used by Frank-Wolfe algorithms:

        f(x) = -F(x)
             = -x^T A x - 0.5 ||x||_2^2

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix of the graph.
    x : np.ndarray
        Current iterate.
    check_input : bool
        If True, validate A and x before computing the objective.

    Returns
    -------
    float
        Value of f(x).
    """
    return -max_clique_objective(A, x, check_input=check_input)


def gradient(A, x, check_input=False):
    """
    Compute the gradient of the minimization objective:

        f(x) = -x^T A x - 0.5 ||x||_2^2

    Since A is symmetric:

        grad f(x) = -2 A x - x

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix of the graph.
    x : np.ndarray
        Current iterate.
    check_input : bool
        If True, validate A and x before computing the gradient.

    Returns
    -------
    np.ndarray
        Gradient vector with shape (n,).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)

    if check_input:
        validate_adjacency_matrix(A)
        validate_vector(A, x)

    return -2.0 * (A @ x) - x


def is_in_simplex(x, tol=1e-8):
    """
    Check whether x belongs to the standard simplex:

        x_i >= 0 for all i
        sum_i x_i = 1

    Parameters
    ----------
    x : np.ndarray
        Vector to check.
    tol : float
        Numerical tolerance.

    Returns
    -------
    bool
        True if x is in the simplex up to tolerance.
    """
    x = np.asarray(x, dtype=float)

    nonnegative = np.all(x >= -tol)
    sums_to_one = abs(np.sum(x) - 1.0) <= tol

    return bool(nonnegative and sums_to_one)


def simplex_uniform_point(n):
    """
    Create the uniform point in the simplex:

        x = (1/n, ..., 1/n)

    This can be used as an initial point.

    Parameters
    ----------
    n : int
        Dimension of the simplex.

    Returns
    -------
    np.ndarray
        Uniform simplex vector.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    return np.ones(n, dtype=float) / n


def support_indices(x, tol=1e-10):
    """
    Return the indices where x is numerically positive.

    In this project, the support of x represents the selected graph vertices.

    Parameters
    ----------
    x : np.ndarray
        Current iterate.
    tol : float
        Values larger than tol are considered active.

    Returns
    -------
    np.ndarray
        Array of active indices.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > tol)[0]


def support_size(x, tol=1e-10):
    """
    Return the number of numerically positive components of x.

    Parameters
    ----------
    x : np.ndarray
        Current iterate.
    tol : float
        Values larger than tol are considered active.

    Returns
    -------
    int
        Number of active components.
    """
    return int(len(support_indices(x, tol=tol)))