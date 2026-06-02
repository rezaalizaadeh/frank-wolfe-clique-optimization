"""
lmo.py

Linear Minimization Oracle and Frank-Wolfe gap utilities.

For the simplex:
    Delta = {x >= 0, sum(x) = 1}

The LMO solves:
    min_s <s, g> subject to s in Delta

The solution is the unit vector e_i where:
    i = argmin_i g_i
"""

import numpy as np

from objective import gradient


def simplex_lmo(g):
    """
    Linear Minimization Oracle over the standard simplex.

    Solves:
        min_s <s, g>
        subject to s in simplex.

    Since the feasible set is the simplex, the minimizer is a vertex:
        s = e_i
    where:
        i = argmin_i g_i

    Parameters
    ----------
    g : np.ndarray
        Gradient vector.

    Returns
    -------
    s : np.ndarray
        Simplex vertex e_i.
    i : int
        Index of the selected vertex.
    """
    g = np.asarray(g, dtype=float)

    if g.ndim != 1:
        raise ValueError("g must be a 1D vector.")

    n = g.shape[0]
    i = int(np.argmin(g))

    s = np.zeros(n, dtype=float)
    s[i] = 1.0

    return s, i


def active_set_lmo(g, active_indices):
    """
    Linear oracle restricted to the current active set.

    This is used by Away-Step FW and Pairwise FW to choose the away atom.

    It solves:
        max_v <v, g>
        over active simplex vertices v = e_j, j in active set.

    Since <e_j, g> = g_j, this returns:
        j = argmax_{j in active set} g_j

    Parameters
    ----------
    g : np.ndarray
        Gradient vector of the minimization objective.
    active_indices : array-like
        Indices with positive weight in the current iterate x.

    Returns
    -------
    v : np.ndarray
        Away atom e_j.
    j : int
        Index of the away atom.
    """
    g = np.asarray(g, dtype=float)
    active_indices = np.asarray(active_indices, dtype=int)

    if g.ndim != 1:
        raise ValueError("g must be a 1D vector.")

    if active_indices.ndim != 1:
        raise ValueError("active_indices must be a 1D array.")

    if len(active_indices) == 0:
        raise ValueError("active set cannot be empty.")

    if np.any(active_indices < 0) or np.any(active_indices >= len(g)):
        raise ValueError("active_indices contains invalid indices.")

    local_position = int(np.argmax(g[active_indices]))
    j = int(active_indices[local_position])

    v = np.zeros_like(g, dtype=float)
    v[j] = 1.0

    return v, j


def frank_wolfe_gap(A, x):
    """
    Compute the Frank-Wolfe gap for the minimization problem.

    General FW gap:
        G(x) = max_s -grad f(x)^T (s - x)

    If s is the LMO solution:
        s = argmin_z <z, grad f(x)>

    then:
        G(x) = <x - s, grad f(x)>

    This value is nonnegative up to numerical tolerance.
    It is used as a stopping criterion.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    x : np.ndarray
        Current iterate.

    Returns
    -------
    gap : float
        Frank-Wolfe gap.
    s : np.ndarray
        FW atom selected by the simplex LMO.
    i : int
        Index of the FW atom.
    """
    g = gradient(A, x)
    s, i = simplex_lmo(g)

    gap = float((x - s) @ g)

    # Numerical protection: tiny negative values may appear from floating point error.
    if gap < 0 and abs(gap) < 1e-12:
        gap = 0.0

    return gap, s, i


def pairwise_gap(A, x, active_indices):
    """
    Compute the pairwise Frank-Wolfe directional gap.

    Pairwise FW direction:
        d = s - v

    where:
        s = FW atom from simplex LMO
        v = away atom from active set

    Descent measure for minimization:
        -grad f(x)^T d

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    x : np.ndarray
        Current iterate.
    active_indices : array-like
        Active indices of x.

    Returns
    -------
    gap : float
        Pairwise directional gap.
    s : np.ndarray
        FW atom.
    i : int
        FW atom index.
    v : np.ndarray
        Away atom.
    j : int
        Away atom index.
    """
    g = gradient(A, x)

    s, i = simplex_lmo(g)
    v, j = active_set_lmo(g, active_indices)

    d = s - v
    gap = float(-g @ d)

    if gap < 0 and abs(gap) < 1e-12:
        gap = 0.0

    return gap, s, i, v, j


def unit_vector(n, index):
    """
    Create the standard basis vector e_index.

    Parameters
    ----------
    n : int
        Dimension.
    index : int
        Index of the unit vector.

    Returns
    -------
    np.ndarray
        Vector e_index.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if index < 0 or index >= n:
        raise ValueError("index out of range.")

    e = np.zeros(n, dtype=float)
    e[index] = 1.0

    return e