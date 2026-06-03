"""
Problem (Equation 31 of the paper):
    max  F(x) = x^T A x + (1/2) ||x||_2^2
    s.t. x in Delta_n (standard simplex)

Frank-Wolfe is written for minimization, so internally we minimize
    f(x) = -F(x).

"""

import numpy as np
import matplotlib.pyplot as plt
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


# ============================================================
# PROJECT PATHS
# ============================================================
# Resolved relative to this file, so the script works no matter the current
# working directory. Layout:  Project/code/frank_wolfe.py
#                             Project/data/*.mtx
#                             Project/plots/*.png
HERE         = os.path.dirname(os.path.abspath(__file__))   # .../Project/code
PROJECT_ROOT = os.path.dirname(HERE)                        # .../Project
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "plots")


# ============================================================
# SECTION 1: CLIQUE / SIMPLEX HELPERS
# ============================================================

def is_clique(A, vertices):
    """
    Returns True if the given vertices form a valid clique in the graph
    with adjacency matrix A (symmetric, 0 on diagonal).
    """
    vertices = np.asarray(vertices, dtype=int)

    if len(vertices) <= 1:
        return True

    submatrix = A[np.ix_(vertices, vertices)]
    k = len(vertices)
    # In a clique of size k there are k*(k-1) directed edges (A symmetric).
    return int(np.sum(submatrix)) == k * (k - 1)


def clique_edge_density(A, vertices):
    """
    Edge density inside the selected support:
        density = existing directed off-diagonal edges / possible ones.
    Density = 1 means the vertices form a clique.
    """
    vertices = np.asarray(vertices, dtype=int)

    if len(vertices) <= 1:
        return 1.0

    submatrix = A[np.ix_(vertices, vertices)]
    k = len(vertices)
    possible_directed_edges = k * (k - 1)
    existing_directed_edges = np.sum(submatrix)

    return float(existing_directed_edges / possible_directed_edges)


def extract_valid_clique(candidate_nodes, A):
    """
    Fallback (Frank-Wolfe specific). From a set of candidate nodes, extracts the
    largest valid clique by iteratively removing the least-connected node.

    Classic FW does not zero out coordinates, so its thresholded support may not
    be a clean clique; this guarantees the returned node set is a true clique.
    """
    nodes = list(candidate_nodes)
    while len(nodes) > 1:
        if is_clique(A, np.array(nodes)):
            return np.array(nodes)
        # Remove the node with fewest edges inside the candidate set.
        degrees = [sum(A[v, m] for m in nodes if m != v) for v in nodes]
        nodes.pop(int(np.argmin(degrees)))
    return np.array(nodes, dtype=int) if nodes else np.array([], dtype=int)


def random_simplex_point(n, rng=None):
    """
    Random point in the simplex via normalized positive exponentials
    (equivalent to a Dirichlet(1, ..., 1) draw).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    y = rng.exponential(scale=1.0, size=n)
    return y / np.sum(y)


def random_vertex_point(n, rng=None):
    """
    Random simplex vertex e_i (atom-based initialization).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    index = int(rng.integers(0, n))
    x = np.zeros(n, dtype=float)
    x[index] = 1.0
    return x


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
        "mean_clique_size": float(np.mean(clique_sizes)),
        "std_clique_size": float(np.std(clique_sizes)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "num_clique_outputs": int(np.sum(clique_flags)),
        "success_rate_clique_output": float(np.mean(clique_flags)),
    }

    return summary


# ============================================================
# SECTION 4: RESULTS - TEXT SUMMARY, TABLE AND PLOTS
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


def plot_results(all_summaries, datasets, save_path="frank_wolfe_results.png"):
    """
    Three plots:
      1. Convergence curve F(x) (best run per dataset)
      2. Box plot of clique sizes across all starts
      3. Bar chart: mean and best quality vs best known (%)
    """
    existing = [ds for ds in datasets if ds in all_summaries]
    if not existing:
        print("[WARNING] No results to plot.")
        return
    colors = ['steelblue', 'darkorange', 'forestgreen']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Frank-Wolfe - L2 Regularized Max-Clique (DIMACS Instances)',
        fontsize=13, fontweight='bold'
    )

    # Plot 1: Convergence (F(x) of the best run per dataset)
    ax = axes[0]
    for ds, color in zip(existing, colors):
        hist  = all_summaries[ds]['best_result']['objective_history']
        best  = all_summaries[ds]['best_clique_size']
        short = ds.replace('.mtx', '')
        ax.plot(hist, label=f"{short}  (best clique={best})", color=color, linewidth=1.8)

    ax.set_title('Convergence (best run per dataset)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective value  F(x) = x^T A x + 0.5||x||^2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot of clique sizes
    ax     = axes[1]
    data   = [np.array([r["final_support_size"] for r in all_summaries[ds]['all_results']])
              for ds in existing]
    labels = [ds.replace('.mtx', '') for ds in existing]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for i, (ds, color) in enumerate(zip(existing, colors), start=1):
        bk = BEST_KNOWN.get(ds)
        if bk:
            ax.axhline(y=bk, color=color, linestyle='--', linewidth=1.2, alpha=0.7)
            ax.plot(i, bk, '*', color=color, markersize=14,
                    label=f"{ds.replace('.mtx','')} best known={bk}")

    ax.set_title('Clique Size Distribution\n(* = Best known  |  --- = Best known)')
    ax.set_ylabel('Clique size')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 3: Quality bar chart
    ax       = axes[2]
    ds_short = [ds.replace('.mtx', '') for ds in existing]
    q_mean   = [np.mean([r["final_support_size"] for r in all_summaries[ds]['all_results']]) /
                BEST_KNOWN.get(ds, 1) * 100 for ds in existing]
    q_max    = [np.max([r["final_support_size"] for r in all_summaries[ds]['all_results']]) /
                BEST_KNOWN.get(ds, 1) * 100 for ds in existing]
    x_pos    = np.arange(len(ds_short))

    ax.bar(x_pos - 0.2, q_mean, 0.38, label='Mean quality',
           color=colors[:len(existing)], alpha=0.55)
    ax.bar(x_pos + 0.2, q_max,  0.38, label='Best quality',
           color=colors[:len(existing)], alpha=1.0)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Best known (100 %)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_short)
    ax.set_title('Solution Quality vs Best Known')
    ax.set_ylabel('Quality (%)')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved to '{save_path}'")
    plt.show()


# ============================================================
# SECTION 5: MAIN
# ============================================================

def run_experiments(num_starts=100, max_iter=1000, tol=1e-6, seed=42,
                    start_mode="uniform_random"):
    """
    Run the full multistart Frank-Wolfe experiment on the 3 DIMACS datasets.
    Prints a per-dataset text summary (same format as pairwise_fw.py), then the
    results table, and generates all plots.
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

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_save_path = os.path.join(PLOTS_DIR, "frank_wolfe_results.png")
    plot_results(all_summaries, datasets, save_path=plot_save_path)

    return all_summaries


if __name__ == "__main__":
    results = run_experiments(num_starts=100)
