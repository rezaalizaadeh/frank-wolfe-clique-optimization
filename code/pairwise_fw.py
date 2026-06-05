"""
pairwise_fw.py

Pairwise Frank-Wolfe algorithm for the L2-regularized maximum clique problem.

Original maximization problem:

    F(x) = x^T A x + 0.5 ||x||_2^2

subject to x in the simplex.

We implement the equivalent minimization problem:

    f(x) = -F(x)

The Pairwise Frank-Wolfe update follows:

    s = FW atom
    v = away atom from the active set
    d = s - v
    gamma_max = alpha_v

In the simplex, the atom coefficients alpha_v are exactly the entries of x.
Therefore:

    gamma_max = x[away_index]
"""

import os
import time
import numpy as np

from graph_loader import load_mtx_graph
from objective import (
    minimization_objective,
    max_clique_objective,
    gradient,
    simplex_uniform_point,
    support_indices,
    support_size,
    is_in_simplex,
)
from lmo import active_set_lmo, frank_wolfe_gap
from line_search import exact_line_search
from utils import extract_valid_clique


# ============================================================
# PROJECT PATHS
# ============================================================
# Resolved relative to this file, so the script works no matter the current
# working directory. Layout:  Project/code/pairwise_fw.py
#                             Project/data/*.mtx
HERE         = os.path.dirname(os.path.abspath(__file__))   # .../Project/code
PROJECT_ROOT = os.path.dirname(HERE)                        # .../Project
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")


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

    This matches the atom-based initialization used in many FW descriptions.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    index = int(rng.integers(0, n))
    x = np.zeros(n, dtype=float)
    x[index] = 1.0

    return x


def pairwise_frank_wolfe(A, x0=None, max_iter=1000, tol=1e-6, active_tol=1e-10):
    """
    Run Pairwise Frank-Wolfe on the L2-regularized maximum clique objective.

    Pairwise Frank-Wolfe transfers weight from a bad active atom v
    to a good Frank-Wolfe atom s.

    At iteration k:

        s = argmin_{z in simplex} <z, grad f(x)>
        v = argmax_{z in active set} <z, grad f(x)>
        d = s - v
        gamma_max = alpha_v = x[v_index]
        x_new = x + gamma * d

    Parameters
    ----------
    A : np.ndarray
        Symmetric adjacency matrix of the graph.
    x0 : np.ndarray or None
        Initial point in the simplex. If None, use the uniform simplex point.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stopping tolerance based on the Frank-Wolfe gap.
    active_tol : float
        Tolerance for deciding whether a component of x is active.

    Returns
    -------
    dict
        Dictionary containing the final solution and convergence history.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = simplex_uniform_point(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    if not is_in_simplex(x):
        raise ValueError("Initial point x0 must belong to the simplex.")

    objective_history = []
    minimization_history = []
    gap_history = []
    support_size_history = []
    step_size_history = []
    fw_atom_history = []
    away_atom_history = []
    step_type_history = []

    good_step_count = 0
    drop_step_count = 0
    swap_step_count = 0

    start_time = time.time()

    for iteration in range(max_iter):
        # Evaluate current objective values.
        F_value = max_clique_objective(A, x)
        f_value = minimization_objective(A, x)

        # FW gap and FW atom.
        gap, s, fw_index = frank_wolfe_gap(A, x)

        objective_history.append(F_value)
        minimization_history.append(f_value)
        gap_history.append(gap)
        support_size_history.append(support_size(x, tol=active_tol))
        fw_atom_history.append(fw_index)

        # Stopping criterion.
        if gap <= tol:
            break

        # Gradient of minimization objective.
        g = gradient(A, x)

        # Active set: indices with positive weight.
        active = support_indices(x, tol=active_tol)

        # Away atom: active vertex with largest gradient component.
        v, away_index = active_set_lmo(g, active)
        away_atom_history.append(away_index)

        # Pairwise direction: move mass from away atom to FW atom.
        d = s - v

        # In the simplex, the coefficient alpha_v of atom e_v is exactly x[away_index].
        # Therefore the maximum feasible pairwise step is gamma_max = alpha_v.
        gamma_max = float(x[away_index])

        # Exact line search over [0, gamma_max].
        gamma = exact_line_search(A, x, d, gamma_max)
        step_size_history.append(gamma)

        # Classify the step following the terminology in FW variants literature.
        # For Pairwise FW:
        # - good step: gamma < gamma_max
        # - drop step: gamma = gamma_max and FW atom was already active
        # - swap step: gamma = gamma_max and FW atom was not active
        fw_was_active = x[fw_index] > active_tol

        if abs(gamma - gamma_max) <= 1e-12:
            if fw_was_active:
                step_type = "drop"
                drop_step_count += 1
            else:
                step_type = "swap"
                swap_step_count += 1
        else:
            step_type = "good"
            good_step_count += 1

        step_type_history.append(step_type)

        # Update iterate.
        x = x + gamma * d

        # Numerical cleanup.
        x[np.abs(x) < active_tol] = 0.0

        # Re-normalize to avoid tiny floating point drift.
        total_mass = np.sum(x)
        if total_mass <= 0:
            raise RuntimeError("Numerical error: simplex mass became nonpositive.")

        x = x / total_mass

        # Safety check.
        if not is_in_simplex(x, tol=1e-7):
            raise RuntimeError("Numerical error: iterate left the simplex.")

    runtime = time.time() - start_time

    # Unified clique extraction (same as frank_wolfe.py) so the three algorithms
    # report clique size the same way and the comparison is fair.
    raw_support_size = support_size(x, tol=active_tol)   # diagnostic: sparsity of the iterate
    threshold = 1.0 / (2 * n)
    clique_nodes = support_indices(x, tol=threshold)
    if not is_clique(A, clique_nodes):
        clique_nodes = extract_valid_clique(clique_nodes, A)

    selected_vertices = clique_nodes
    final_is_clique = is_clique(A, clique_nodes)
    final_clique_density = clique_edge_density(A, clique_nodes)

    result = {
        "x_final": x,
        "objective_history": objective_history,
        "minimization_history": minimization_history,
        "gap_history": gap_history,
        "support_size_history": support_size_history,
        "step_size_history": step_size_history,
        "fw_atom_history": fw_atom_history,
        "away_atom_history": away_atom_history,
        "step_type_history": step_type_history,
        "runtime": runtime,
        "iterations": len(objective_history),
        "selected_vertices": selected_vertices,
        "final_objective": max_clique_objective(A, x),
        "final_minimization_objective": minimization_objective(A, x),
        "final_gap": gap_history[-1],
        "final_support_size": len(clique_nodes),
        # Raw thresholded support at convergence (kept as a sparsity diagnostic).
        "final_raw_support_size": raw_support_size,
        "final_is_clique": final_is_clique,
        "final_clique_density": final_clique_density,
        "good_step_count": good_step_count,
        "drop_step_count": drop_step_count,
        "swap_step_count": swap_step_count,
    }

    return result


def multistart_pairwise_frank_wolfe(
    A,
    num_starts=20,
    max_iter=1000,
    tol=1e-6,
    active_tol=1e-10,
    seed=42,
    start_mode="uniform_random",
):
    """
    Run Pairwise Frank-Wolfe from multiple starting points.

    This is useful because the problem is non-convex. Different starting points
    can converge to different local maximizers / maximal cliques.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    num_starts : int
        Number of runs.
    max_iter : int
        Maximum iterations per run.
    tol : float
        FW gap tolerance.
    active_tol : float
        Support tolerance.
    seed : int
        Random seed for reproducibility.
    start_mode : str
        "uniform_random":
            first start is uniform, remaining starts are random simplex points.
        "mixed":
            first start is uniform, then random vertex starts for half of the runs,
            then random simplex starts for the remaining runs.
        "vertices":
            all starts are random simplex vertices.
        "random":
            all starts are random simplex points.

    Returns
    -------
    dict
        Summary of all runs and the best run.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if num_starts <= 0:
        raise ValueError("num_starts must be positive.")

    valid_modes = {"uniform_random", "mixed", "vertices", "random"}
    if start_mode not in valid_modes:
        raise ValueError(f"start_mode must be one of {valid_modes}.")

    rng = np.random.default_rng(seed)
    all_results = []

    for start_id in range(num_starts):
        if start_mode == "uniform_random":
            if start_id == 0:
                x0 = simplex_uniform_point(n)
                start_type = "uniform"
            else:
                x0 = random_simplex_point(n, rng=rng)
                start_type = "random_simplex"

        elif start_mode == "mixed":
            if start_id == 0:
                x0 = simplex_uniform_point(n)
                start_type = "uniform"
            elif start_id <= num_starts // 2:
                x0 = random_vertex_point(n, rng=rng)
                start_type = "random_vertex"
            else:
                x0 = random_simplex_point(n, rng=rng)
                start_type = "random_simplex"

        elif start_mode == "vertices":
            x0 = random_vertex_point(n, rng=rng)
            start_type = "random_vertex"

        else:  # start_mode == "random"
            x0 = random_simplex_point(n, rng=rng)
            start_type = "random_simplex"

        result = pairwise_frank_wolfe(
            A,
            x0=x0,
            max_iter=max_iter,
            tol=tol,
            active_tol=active_tol,
        )

        result["start_id"] = start_id
        result["start_type"] = start_type

        all_results.append(result)

    # Best result: prioritize clique size, then objective.
    best_result = max(
        all_results,
        key=lambda r: (r["final_support_size"], r["final_objective"]),
    )

    clique_sizes = np.array([r["final_support_size"] for r in all_results], dtype=float)
    raw_support_sizes = np.array([r["final_raw_support_size"] for r in all_results], dtype=float)
    objectives = np.array([r["final_objective"] for r in all_results], dtype=float)
    runtimes = np.array([r["runtime"] for r in all_results], dtype=float)
    clique_flags = np.array([r["final_is_clique"] for r in all_results], dtype=bool)
    good_steps = np.array([r["good_step_count"] for r in all_results], dtype=float)
    drop_steps = np.array([r["drop_step_count"] for r in all_results], dtype=float)
    swap_steps = np.array([r["swap_step_count"] for r in all_results], dtype=float)

    summary = {
        "all_results": all_results,
        "best_result": best_result,
        "num_starts": num_starts,
        "start_mode": start_mode,
        "n_nodes": n,
        "best_clique_size": best_result["final_support_size"],
        "best_objective": best_result["final_objective"],
        "best_vertices": best_result["selected_vertices"],
        "mean_clique_size": float(np.mean(clique_sizes)),
        "std_clique_size": float(np.std(clique_sizes)),
        "mean_support_size_all_runs": float(np.mean(raw_support_sizes)),
        "std_support_size_all_runs": float(np.std(raw_support_sizes)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "num_clique_outputs": int(np.sum(clique_flags)),
        "success_rate_clique_output": float(np.mean(clique_flags)),
        "mean_good_steps": float(np.mean(good_steps)),
        "mean_drop_steps": float(np.mean(drop_steps)),
        "mean_swap_steps": float(np.mean(swap_steps)),
    }

    return summary


# ============================================================
# RESULTS - TEXT SUMMARY AND TABLE
# ============================================================

# Best known clique sizes for the selected DIMACS instances (from Table 1 of the paper).
# Keys MUST match the actual .mtx filenames in the data/ directory.
BEST_KNOWN = {
    'C125-9.mtx':     34,
    'brock200-2.mtx': 12,
    'keller4.mtx':    11,
}


def print_summary(dataset_name, summary):
    """
    Print the multistart summary. Common block matches frank_wolfe.py; the
    pairwise-specific step statistics (good/drop/swap) are appended at the end.
    """
    best = summary["best_result"]

    print(f"Multistart Pairwise Frank-Wolfe test on {dataset_name}")
    print("------------------------------------------------")
    print("Number of starts:", summary["num_starts"])
    print("Start mode:", summary["start_mode"])
    print("Best clique size:", summary["best_clique_size"])
    print("Best objective:", summary["best_objective"])
    print("Best vertices:", summary["best_vertices"])
    print("Mean clique size:", summary["mean_clique_size"])
    print("Std clique size:", summary["std_clique_size"])
    print("Mean objective:", summary["mean_objective"])
    print("Std objective:", summary["std_objective"])
    print("Mean runtime:", summary["mean_runtime"])
    print("Total runtime:", summary["total_runtime"])
    print("Clique output success rate:", summary["success_rate_clique_output"])
    print("Mean good steps:", summary["mean_good_steps"])
    print("Mean drop steps:", summary["mean_drop_steps"])
    print("Mean swap steps:", summary["mean_swap_steps"])

    print("\nBest run details")
    print("----------------")
    print("Start ID:", best["start_id"])
    print("Start type:", best["start_type"])
    print("Iterations:", best["iterations"])
    print("Final FW gap:", best["final_gap"])
    print("Final support is clique:", best["final_is_clique"])
    print("Final clique density:", best["final_clique_density"])
    print("Good steps:", best["good_step_count"])
    print("Drop steps:", best["drop_step_count"])
    print("Swap steps:", best["swap_step_count"])


def print_results_table(all_summaries, datasets):
    """
    Compact results table across datasets: Max, Mean, Std, Quality %, Avg Time.
    Mirrors the structure of Table 2 in the paper.
    """
    col = (f"\n{'Dataset':<17} | {'Nodes':<6} | {'Best Known':<11} | "
           f"{'Max':<5} | {'Mean':>7} | {'Std':>5} | "
           f"{'Quality %':>10} | {'Avg Time (s)':>12}")
    sep = '-' * len(col)

    print(sep)
    print(col)
    print(sep)

    for ds in datasets:
        if ds not in all_summaries:
            continue
        summ  = all_summaries[ds]
        n     = summ["n_nodes"]
        bk    = BEST_KNOWN.get(ds, None)
        sizes = np.array([r["final_support_size"] for r in summ["all_results"]])
        times = np.array([r["runtime"] for r in summ["all_results"]])
        q     = f"{sizes.max() / bk * 100:.1f}%" if bk else "N/A"

        print(f"{ds:<17} | {n:<6} | {str(bk):<11} | "
              f"{sizes.max():<5} | {sizes.mean():>7.2f} | {sizes.std():>5.2f} | "
              f"{q:>10} | {times.mean():>12.4f}")

    print(sep + "\n")


# ============================================================
# MAIN
# ============================================================

def run_experiments(num_starts=100, max_iter=1000, tol=1e-6, seed=42,
                    start_mode="uniform_random"):
    """
    Run the full multistart Pairwise Frank-Wolfe experiment on the 3 DIMACS
    datasets. Prints a per-dataset text summary, then the results table.
    Plot/CSV generation lives in a separate script.
    """
    datasets      = list(BEST_KNOWN.keys())
    all_summaries = {}

    for ds in datasets:
        dataset_path = os.path.join(DATA_DIR, ds)
        if not os.path.exists(dataset_path):
            print(f"[WARNING] File not found: '{dataset_path}'. "
                  f"Place the .mtx file in the data/ directory and re-run.")
            continue

        A = load_mtx_graph(dataset_path)

        print(f"\n[INFO] Dataset: {ds}  |  Nodes: {A.shape[0]}  |  "
              f"Running {num_starts} starts...", flush=True)

        summary = multistart_pairwise_frank_wolfe(
            A, num_starts=num_starts, max_iter=max_iter, tol=tol,
            seed=seed, start_mode=start_mode,
        )
        all_summaries[ds] = summary

        print()
        print_summary(ds, summary)

    print()
    print_results_table(all_summaries, datasets)

    return all_summaries


if __name__ == "__main__":
    results = run_experiments(num_starts=100)