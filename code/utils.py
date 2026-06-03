"""
utils.py

Shared helper functions for the Frank-Wolfe maximum clique project.
"""

import numpy as np


def is_clique(A, vertices):
    """
    Check whether the selected vertices form a clique.

    A set of vertices is a clique if every pair of different vertices
    is connected by an edge.
    """
    vertices = np.asarray(vertices, dtype=int)

    if len(vertices) <= 1:
        return True

    submatrix = A[np.ix_(vertices, vertices)]
    k = len(vertices)

    required_directed_edges = k * (k - 1)
    actual_directed_edges = int(np.sum(submatrix))

    return actual_directed_edges == required_directed_edges


def clique_edge_density(A, vertices):
    """
    Compute edge density inside the selected support.

    Density = existing directed off-diagonal edges / possible directed off-diagonal edges.

    If density = 1, the selected vertices form a clique.
    """
    vertices = np.asarray(vertices, dtype=int)

    if len(vertices) <= 1:
        return 1.0

    submatrix = A[np.ix_(vertices, vertices)]

    k = len(vertices)
    possible_directed_edges = k * (k - 1)
    existing_directed_edges = np.sum(submatrix)

    return float(existing_directed_edges / possible_directed_edges)


def random_simplex_point(n, rng=None):
    """
    Generate a random point in the simplex.

    Positive exponential random variables normalized by their sum give
    a random probability vector.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    y = rng.exponential(scale=1.0, size=n)
    return y / np.sum(y)


def random_vertex_point(n, rng=None):
    """
    Generate a random simplex vertex e_i.

    This is useful for FW-type algorithms because the simplex atoms are
    the standard unit vectors.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    index = int(rng.integers(0, n))
    x = np.zeros(n, dtype=float)
    x[index] = 1.0

    return x