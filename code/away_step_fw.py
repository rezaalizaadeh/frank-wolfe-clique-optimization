"""
away_step_fw.py

Away-Step Frank-Wolfe algorithm for the L2-regularized maximum clique problem.

Original maximization problem:

    F(x) = x^T A x + 0.5 ||x||_2^2

subject to x in the simplex.

We implement the equivalent minimization problem:

    f(x) = -F(x)

Away-Step Frank-Wolfe chooses between:

    FW direction:      d_FW = s - x
    Away direction:    d_A  = x - v

where:
    s = Frank-Wolfe atom
    v = away atom from the active set

For an away step, the maximum feasible step is:

    gamma_max = alpha_v / (1 - alpha_v)

In the simplex, alpha_v = x[away_index].
"""

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
from utils import (
    is_clique,
    clique_edge_density,
    random_simplex_point,
    random_vertex_point,
)


def away_step_frank_wolfe(A, x0=None, max_iter=1000, tol=1e-6, active_tol=1e-10):
    """
    Run Away-Step Frank-Wolfe on the L2-regularized maximum clique objective.

    At each iteration:

        1. Compute FW atom:
            s = argmin_{z in simplex} <z, grad f(x)>

        2. Compute away atom:
            v = argmax_{z in active set} <z, grad f(x)>

        3. Compare directions:
            d_FW = s - x
            d_A  = x - v

        4. Choose the direction with larger descent value.
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

    fw_step_count = 0
    away_step_count = 0
    drop_step_count = 0
    good_step_count = 0

    start_time = time.time()

    for iteration in range(max_iter):
        # Current objective values.
        F_value = max_clique_objective(A, x)
        f_value = minimization_objective(A, x)

        # FW gap and FW atom.
        gap, s, fw_index = frank_wolfe_gap(A, x)

        objective_history.append(F_value)
        minimization_history.append(f_value)
        gap_history.append(gap)
        support_size_history.append(support_size(x, tol=active_tol))
        fw_atom_history.append(fw_index)

        # Stopping criterion based on FW gap.
        if gap <= tol:
            break

        # Gradient of minimization objective.
        g = gradient(A, x)

        # Active set.
        active = support_indices(x, tol=active_tol)

        # Away atom: active vertex with largest gradient component.
        v, away_index = active_set_lmo(g, active)
        away_atom_history.append(away_index)

        # FW direction and away direction.
        d_fw = s - x
        d_away = x - v

        # Descent values:
        # We minimize f, so a direction is good if -<grad, d> is large.
        gap_fw = -float(g @ d_fw)
        gap_away = -float(g @ d_away)

        # Choose between FW step and away step.
        if gap_fw >= gap_away:
            d = d_fw
            gamma_max = 1.0
            step_type = "fw"
            fw_step_count += 1
        else:
            d = d_away

            # In the simplex, alpha_v = x[away_index].
            alpha_v = float(x[away_index])

            # Away-step maximum feasible step:
            # gamma_max = alpha_v / (1 - alpha_v)
            denominator = 1.0 - alpha_v

            if denominator <= active_tol:
                # This can happen when x is essentially a vertex.
                # In that case, the away direction is numerically unsafe/useless,
                # so we fall back to a FW step.
                d = d_fw
                gamma_max = 1.0
                step_type = "fw"
                fw_step_count += 1
            else:
                gamma_max = alpha_v / denominator
                step_type = "away"
                away_step_count += 1

        # Exact line search over [0, gamma_max].
        gamma = exact_line_search(A, x, d, gamma_max)
        step_size_history.append(gamma)

        # Classify good/drop step.
        # For AFW, a drop step happens when an away step takes gamma_max.
        if step_type == "away" and abs(gamma - gamma_max) <= 1e-12:
            step_type = "drop"
            drop_step_count += 1
        else:
            good_step_count += 1

        step_type_history.append(step_type)

        # Update iterate.
        x = x + gamma * d

        # Numerical cleanup.
        x[np.abs(x) < active_tol] = 0.0

        # Re-normalize to avoid tiny floating-point drift.
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
        "away_atom_history": away_atom_history,
        "step_type_history": step_type_history,
        "runtime": runtime,
        "iterations": len(objective_history),
        "selected_vertices": selected_vertices,
        "final_objective": max_clique_objective(A, x),
        "final_minimization_objective": minimization_objective(A, x),
        "final_gap": gap_history[-1],
        "final_support_size": support_size(x, tol=active_tol),
        "final_is_clique": final_is_clique,
        "final_clique_density": final_clique_density,
        "fw_step_count": fw_step_count,
        "away_step_count": away_step_count,
        "drop_step_count": drop_step_count,
        "good_step_count": good_step_count,
    }

    return result


def multistart_away_step_frank_wolfe(
    A,
    num_starts=20,
    max_iter=1000,
    tol=1e-6,
    active_tol=1e-10,
    seed=42,
    start_mode="mixed",
):
    """
    Run Away-Step Frank-Wolfe from multiple starting points.

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
        Away-Step FW is expected to improve support sparsity compared with
        classical FW, but we still select best results among valid clique outputs first.
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

        result = away_step_frank_wolfe(
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

    support_sizes = np.array([r["final_support_size"] for r in all_results], dtype=float)
    objectives = np.array([r["final_objective"] for r in all_results], dtype=float)
    runtimes = np.array([r["runtime"] for r in all_results], dtype=float)
    clique_flags = np.array([r["final_is_clique"] for r in all_results], dtype=bool)

    fw_steps = np.array([r["fw_step_count"] for r in all_results], dtype=float)
    away_steps = np.array([r["away_step_count"] for r in all_results], dtype=float)
    drop_steps = np.array([r["drop_step_count"] for r in all_results], dtype=float)
    good_steps = np.array([r["good_step_count"] for r in all_results], dtype=float)

    summary = {
        "all_results": all_results,
        "valid_clique_results": valid_clique_results,
        "best_result": best_result,
        "best_result_is_valid_clique": best_result_is_valid_clique,
        "num_starts": num_starts,
        "start_mode": start_mode,
        "best_clique_size": best_result["final_support_size"],
        "best_objective": best_result["final_objective"],
        "best_vertices": best_result["selected_vertices"],
        "mean_clique_size": float(np.mean(support_sizes)),
        "std_clique_size": float(np.std(support_sizes)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "num_clique_outputs": int(np.sum(clique_flags)),
        "success_rate_clique_output": float(np.mean(clique_flags)),
        "mean_fw_steps": float(np.mean(fw_steps)),
        "mean_away_steps": float(np.mean(away_steps)),
        "mean_drop_steps": float(np.mean(drop_steps)),
        "mean_good_steps": float(np.mean(good_steps)),
    }

    return summary


if __name__ == "__main__":
    # Local multistart test.
    # Run from project root:
    #     python3 Code/away_step_fw.py

    dataset_name = "keller4.mtx"
    A = load_mtx_graph(f"Data/{dataset_name}")

    summary = multistart_away_step_frank_wolfe(
        A,
        num_starts=20,
        max_iter=1000,
        tol=1e-6,
        seed=42,
        start_mode="mixed",
    )

    best = summary["best_result"]

    print(f"Multistart Away-Step Frank-Wolfe test on {dataset_name}")
    print("------------------------------------------------")
    print("Number of starts:", summary["num_starts"])
    print("Start mode:", summary["start_mode"])
    print("Best result is valid clique:", summary["best_result_is_valid_clique"])
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
    print("Mean FW steps:", summary["mean_fw_steps"])
    print("Mean away steps:", summary["mean_away_steps"])
    print("Mean drop steps:", summary["mean_drop_steps"])
    print("Mean good steps:", summary["mean_good_steps"])

    print("\nBest run details")
    print("----------------")
    print("Start ID:", best["start_id"])
    print("Start type:", best["start_type"])
    print("Iterations:", best["iterations"])
    print("Final FW gap:", best["final_gap"])
    print("Final support is clique:", best["final_is_clique"])
    print("Final clique density:", best["final_clique_density"])
    print("FW steps:", best["fw_step_count"])
    print("Away steps:", best["away_step_count"])
    print("Drop steps:", best["drop_step_count"])
    print("Good steps:", best["good_step_count"])