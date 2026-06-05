"""
Problem (Equation 31 of the paper):
    max  F(x) = x^T A x + (1/2) ||x||_2^2
    s.t. x in Delta_n (standard simplex)

Frank-Wolfe is written for minimization, so internally we minimize
    f(x) = -F(x).

"""

import numpy as np
import os
import time

from graph_loader import load_mtx_graph
from objective import (
    minimization_objective,
    max_clique_objective,
    simplex_uniform_point,
    support_indices,
    support_size,
    is_in_simplex,
)
from lmo import frank_wolfe_gap
from line_search import exact_line_search
from utils import (
    is_clique,
    clique_edge_density,
    extract_valid_clique,
    random_simplex_point,
    random_vertex_point,
)


# ============================================================
# PROJECT PATHS
# ============================================================
# Resolved relative to this file, so the script works no matter the current
# working directory. Layout:  Project/code/frank_wolfe.py
#                             Project/data/*.mtx
HERE         = os.path.dirname(os.path.abspath(__file__))   # .../Project/code
PROJECT_ROOT = os.path.dirname(HERE)                        # .../Project
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")


# ============================================================
# SECTION 1: CLIQUE / SIMPLEX HELPERS
# ============================================================
# All shared clique/simplex helpers (is_clique, clique_edge_density,
# extract_valid_clique, random_simplex_point, random_vertex_point) now live in
# utils.py and are imported above, so the three algorithms share them.


# ============================================================
# SECTION 2: FRANK-WOLFE ALGORITHM
# ============================================================

def frank_wolfe_max_clique(A, x0=None, max_iter=1000, tol=1e-6, active_tol=1e-10):
    """
    Frank-Wolfe algorithm for the L2 Regularized Max-Clique problem.

    Uses the SHARED modules and the minimization convention f(x) = -F(x).

    At iteration k:
        s = argmin_{z in simplex} <z, grad f(x)>     (FW atom, via frank_wolfe_gap)
        gap = <grad f(x), x - s>                      (stop if gap <= tol)
        d = s - x
        gamma* = argmin_{[0,1]} f(x + gamma*d)        (exact line search, gamma_max=1)
        x_new = x + gamma* * d

    Returns a `result` dict with the same keys as pairwise_fw.py (FW-applicable
    subset; away/drop/swap fields are omitted).
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

    start_time = time.time()

    for _ in range(max_iter):
        # Track F(x) (maximization view) and f(x) = -F(x) (what FW minimizes).
        F_value = max_clique_objective(A, x)
        f_value = minimization_objective(A, x)

        # gradient of f + simplex LMO in one shared call; gap >= 0.
        gap, s, fw_index = frank_wolfe_gap(A, x)

        objective_history.append(F_value)
        minimization_history.append(f_value)
        gap_history.append(gap)
        support_size_history.append(support_size(x, tol=active_tol))
        fw_atom_history.append(fw_index)

        # Stopping criterion.
        if gap <= tol:
            break

        d = s - x
        gamma = exact_line_search(A, x, d, gamma_max=1.0)   # FW step: gamma_max = 1
        step_size_history.append(gamma)

        x = x + gamma * d

    runtime = time.time() - start_time

    # Clique extraction (FW-specific). Theory says clique = supp(x*), but classic
    # FW leaves tiny positive dust on non-clique coordinates, so we threshold and
    # fall back to a guaranteed-valid clique if needed.
    threshold = 1.0 / (2 * n)
    clique_nodes = support_indices(x, tol=threshold)
    if not is_clique(A, clique_nodes):
        clique_nodes = extract_valid_clique(clique_nodes, A)

    result = {
        "x_final": x,
        "objective_history": objective_history,
        "minimization_history": minimization_history,
        "gap_history": gap_history,
        "support_size_history": support_size_history,
        "step_size_history": step_size_history,
        "fw_atom_history": fw_atom_history,
        "runtime": runtime,
        "iterations": len(objective_history),
        "selected_vertices": clique_nodes,
        "final_objective": max_clique_objective(A, x),
        "final_minimization_objective": minimization_objective(A, x),
        "final_gap": gap_history[-1],
        "final_support_size": len(clique_nodes),
        # Raw thresholded support at convergence (classic FW leaves dust, so this
        # can exceed the extracted valid clique size above).
        "final_raw_support_size": support_size_history[-1],
        "final_is_clique": is_clique(A, clique_nodes),
        "final_clique_density": clique_edge_density(A, clique_nodes),
    }

    return result


# ============================================================
# SECTION 3: MULTISTART FRAMEWORK
# ============================================================

def multistart_frank_wolfe(
    A,
    num_starts=100,
    max_iter=1000,
    tol=1e-6,
    active_tol=1e-10,
    seed=42,
    start_mode="uniform_random",
):
    """
    Run Frank-Wolfe from multiple starting points (the problem is non-convex, so
    different starts converge to different maximal cliques). The paper uses 100
    trials per instance (Section 3.3).

    start_mode (same options as pairwise_fw.py):
        "uniform_random": first uniform, rest random simplex points.
        "mixed":          uniform, then random vertices (half), then random simplex.
        "vertices":       all random simplex vertices.
        "random":         all random simplex points.

    Returns a `summary` dict with the same keys as pairwise_fw.py (pairwise-only
    step statistics are omitted).
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

        result = frank_wolfe_max_clique(
            A, x0=x0, max_iter=max_iter, tol=tol, active_tol=active_tol
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

    summary = {
        "all_results": all_results,
        "best_result": best_result,
        "num_starts": num_starts,
        "start_mode": start_mode,
        "n_nodes": n,
        "best_clique_size": best_result["final_support_size"],
        "best_objective": best_result["final_objective"],
        "best_vertices": best_result["selected_vertices"],
        "best_result_is_valid_clique": best_result["final_is_clique"],
        "mean_clique_size": float(np.mean(clique_sizes)),
        "std_clique_size": float(np.std(clique_sizes)),
        # Aliases used by main.py: classic FW reports the extracted valid clique
        # ("valid_clique") separately from the raw thresholded support ("support").
        "mean_valid_clique_size": float(np.mean(clique_sizes)),
        "std_valid_clique_size": float(np.std(clique_sizes)),
        "mean_support_size_all_runs": float(np.mean(raw_support_sizes)),
        "std_support_size_all_runs": float(np.std(raw_support_sizes)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "num_clique_outputs": int(np.sum(clique_flags)),
        "success_rate_clique_output": float(np.mean(clique_flags)),
    }

    return summary


# ============================================================
# SECTION 4: RESULTS - TEXT SUMMARY AND TABLE
# ============================================================

# Best known clique sizes for the selected DIMACS instances (from Table 1 of the paper).
# Keys MUST match the actual .mtx filenames in the working directory.
BEST_KNOWN = {
    'C125-9.mtx':     34,
    'brock200-2.mtx': 12,
    'keller4.mtx':    11,
}


def print_summary(dataset_name, summary):
    """
    Print the multistart summary using the SAME text layout as the __main__ block
    of pairwise_fw.py (pairwise-only step statistics are omitted).
    """
    best = summary["best_result"]

    print(f"Multistart Frank-Wolfe test on {dataset_name}")
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

    print("\nBest run details")
    print("----------------")
    print("Start ID:", best["start_id"])
    print("Start type:", best["start_type"])
    print("Iterations:", best["iterations"])
    print("Final FW gap:", best["final_gap"])
    print("Final support is clique:", best["final_is_clique"])
    print("Final clique density:", best["final_clique_density"])


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
# SECTION 5: MAIN
# ============================================================

def run_experiments(num_starts=100, max_iter=1000, tol=1e-6, seed=42,
                    start_mode="uniform_random"):
    """
    Run the full multistart Frank-Wolfe experiment on the 3 DIMACS datasets.
    Prints a per-dataset text summary (same format as pairwise_fw.py), then the
    results table. Plot/CSV generation lives in a separate script.
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

        summary = multistart_frank_wolfe(
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
