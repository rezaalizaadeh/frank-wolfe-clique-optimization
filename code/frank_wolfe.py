"""
frank_wolfe.py

Classical Frank-Wolfe algorithm for the L2-regularized maximum clique problem.

Original maximization problem:

    F(x) = x^T A x + 0.5 ||x||_2^2

subject to x in the simplex.

We implement the equivalent minimization problem:

    f(x) = -F(x)

Classical Frank-Wolfe uses:

    s = argmin_{z in simplex} <z, grad f(x)>
    d = s - x
    gamma_max = 1

Important:
    Classical FW does not explicitly remove bad active atoms.
    Therefore, final supports can be dense and may not form cliques.
    For this reason, multistart selection prioritizes only valid clique outputs.
"""

import time
import numpy as np

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
    random_simplex_point,
    random_vertex_point,
)


def frank_wolfe(A, x0=None, max_iter=1000, tol=1e-6, active_tol=1e-10):
    """
    Run classical Frank-Wolfe on the L2-regularized maximum clique objective.

    At each iteration:

        s = argmin_{z in simplex} <z, grad f(x)>
        d = s - x
        gamma in [0, 1]
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

    for iteration in range(max_iter):
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

        # Classical Frank-Wolfe direction.
        d = s - x

        # Classical FW step is always limited to [0, 1].
        gamma_max = 1.0
        gamma = exact_line_search(A, x, d, gamma_max)
        step_size_history.append(gamma)

        # Update iterate.
        x = x + gamma * d

        # Numerical cleanup.
        x[np.abs(x) < active_tol] = 0.0

        # Re-normalize to avoid floating-point drift.
        total_mass = np.sum(x)
        if total_mass <= 0:
            raise RuntimeError("Numerical error: simplex mass became nonpositive.")

        x = x / total_mass

        # Safety check.
        if not is_in_simplex(x, tol=1e-7):
            raise RuntimeError("Numerical error: iterate left the simplex.")

    runtime = time.time() - start_time

    selected_vertices = support_indices(x, tol=active_tol)

    final_is_clique = is_clique(A, selected_vertices)
    final_clique_density = clique_edge_density(A, selected_vertices)

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
        "selected_vertices": selected_vertices,
        "final_objective": max_clique_objective(A, x),
        "final_minimization_objective": minimization_objective(A, x),
        "final_gap": gap_history[-1],
        "final_support_size": support_size(x, tol=active_tol),
        "final_is_clique": final_is_clique,
        "final_clique_density": final_clique_density,
    }

    return result


def multistart_frank_wolfe(
    A,
    num_starts=20,
    max_iter=1000,
    tol=1e-6,
    active_tol=1e-10,
    seed=42,
    start_mode="mixed",
):
    """
    Run classical Frank-Wolfe from multiple starting points.

    start_mode options:
        "uniform_random":
            first start is uniform, remaining starts are random simplex points.
        "mixed":
            first start is uniform, then random vertex starts for half the runs,
            then random simplex starts.
        "vertices":
            all starts are random simplex vertices.
        "random":
            all starts are random simplex points.

    Important:
        Classical FW can return dense supports that are not cliques.
        Therefore, the best result is selected among valid clique outputs first.
        If no valid clique output exists, it falls back to best objective.
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

        result = frank_wolfe(
            A,
            x0=x0,
            max_iter=max_iter,
            tol=tol,
            active_tol=active_tol,
        )

        result["start_id"] = start_id
        result["start_type"] = start_type

        all_results.append(result)

    valid_clique_results = [r for r in all_results if r["final_is_clique"]]

    if len(valid_clique_results) > 0:
        best_result = max(
            valid_clique_results,
            key=lambda r: (r["final_support_size"], r["final_objective"]),
        )
        best_result_is_valid_clique = True
    else:
        best_result = max(
            all_results,
            key=lambda r: r["final_objective"],
        )
        best_result_is_valid_clique = False

    support_sizes_all = np.array([r["final_support_size"] for r in all_results], dtype=float)
    objectives = np.array([r["final_objective"] for r in all_results], dtype=float)
    runtimes = np.array([r["runtime"] for r in all_results], dtype=float)
    clique_flags = np.array([r["final_is_clique"] for r in all_results], dtype=bool)

    if len(valid_clique_results) > 0:
        valid_clique_sizes = np.array(
            [r["final_support_size"] for r in valid_clique_results],
            dtype=float,
        )
        mean_valid_clique_size = float(np.mean(valid_clique_sizes))
        std_valid_clique_size = float(np.std(valid_clique_sizes))
    else:
        mean_valid_clique_size = 0.0
        std_valid_clique_size = 0.0

    summary = {
        "all_results": all_results,
        "valid_clique_results": valid_clique_results,
        "best_result": best_result,
        "best_result_is_valid_clique": best_result_is_valid_clique,
        "num_starts": num_starts,
        "start_mode": start_mode,

        # Best valid clique result if available.
        # If no valid clique exists, this refers to the best-objective support.
        "best_clique_size": best_result["final_support_size"],
        "best_objective": best_result["final_objective"],
        "best_vertices": best_result["selected_vertices"],

        # Statistics over all runs.
        "mean_support_size_all_runs": float(np.mean(support_sizes_all)),
        "std_support_size_all_runs": float(np.std(support_sizes_all)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),

        # Valid clique statistics.
        "num_clique_outputs": int(np.sum(clique_flags)),
        "success_rate_clique_output": float(np.mean(clique_flags)),
        "mean_valid_clique_size": mean_valid_clique_size,
        "std_valid_clique_size": std_valid_clique_size,
    }

    return summary


if __name__ == "__main__":
    # Local multistart test.
    # Run from project root:
    #     python3 Code/frank_wolfe.py

    dataset_name = "C125-9.mtx"
    A = load_mtx_graph(f"Data/{dataset_name}")

    summary = multistart_frank_wolfe(
        A,
        num_starts=20,
        max_iter=1000,
        tol=1e-6,
        seed=42,
        start_mode="mixed",
    )

    best = summary["best_result"]

    print(f"Multistart Classical Frank-Wolfe test on {dataset_name}")
    print("------------------------------------------------")
    print("Number of starts:", summary["num_starts"])
    print("Start mode:", summary["start_mode"])
    print("Best result is valid clique:", summary["best_result_is_valid_clique"])
    print("Best valid clique/support size:", summary["best_clique_size"])
    print("Best objective:", summary["best_objective"])
    print("Best vertices:", summary["best_vertices"])

    print("\nAll-run statistics")
    print("------------------")
    print("Mean support size, all runs:", summary["mean_support_size_all_runs"])
    print("Std support size, all runs:", summary["std_support_size_all_runs"])
    print("Mean objective:", summary["mean_objective"])
    print("Std objective:", summary["std_objective"])
    print("Mean runtime:", summary["mean_runtime"])
    print("Total runtime:", summary["total_runtime"])

    print("\nClique-output statistics")
    print("------------------------")
    print("Number of valid clique outputs:", summary["num_clique_outputs"])
    print("Clique output success rate:", summary["success_rate_clique_output"])
    print("Mean valid clique size:", summary["mean_valid_clique_size"])
    print("Std valid clique size:", summary["std_valid_clique_size"])

    print("\nBest run details")
    print("----------------")
    print("Start ID:", best["start_id"])
    print("Start type:", best["start_type"])
    print("Iterations:", best["iterations"])
    print("Final FW gap:", best["final_gap"])
    print("Final support is clique:", best["final_is_clique"])
    print("Final clique density:", best["final_clique_density"])