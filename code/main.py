"""
main.py

Run Classical Frank-Wolfe, Away-Step Frank-Wolfe, and Pairwise Frank-Wolfe
on the selected maximum clique benchmark datasets.

Run from project root:

    python3 Code/main.py

Expected folder structure:

    Optimization/
    ├── Code/
    │   ├── main.py
    │   ├── graph_loader.py
    │   ├── objective.py
    │   ├── lmo.py
    │   ├── line_search.py
    │   ├── utils.py
    │   ├── frank_wolfe.py
    │   ├── away_step_fw.py
    │   └── pairwise_fw.py
    ├── Data/
    │   ├── C125-9.mtx
    │   ├── brock200-2.mtx
    │   └── keller4.mtx
    └── Results/
        └── results_summary.csv
"""

import os
import csv
import time

from graph_loader import load_mtx_graph
from frank_wolfe import multistart_frank_wolfe
from away_step_fw import multistart_away_step_frank_wolfe
from pairwise_fw import multistart_pairwise_frank_wolfe


DATASETS = [
    {
        "name": "C125-9.mtx",
        "best_known_clique": 34,
    },
    {
        "name": "brock200-2.mtx",
        "best_known_clique": 12,
    },
    {
        "name": "keller4.mtx",
        "best_known_clique": 11,
    },
]


ALGORITHMS = [
    {
        "name": "Classical FW",
        "function": multistart_frank_wolfe,
    },
    {
        "name": "Away-Step FW",
        "function": multistart_away_step_frank_wolfe,
    },
    {
        "name": "Pairwise FW",
        "function": multistart_pairwise_frank_wolfe,
    },
]


NUM_STARTS = 20
MAX_ITER = 1000
TOL = 1e-6
SEED = 42
START_MODE = "mixed"

DATA_DIR = "Data"
RESULTS_DIR = "Results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")


def run_experiment():
    """
    Run all algorithms on all datasets and save a summary CSV.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_rows = []

    total_start_time = time.time()

    for dataset_info in DATASETS:
        dataset_name = dataset_info["name"]
        best_known = dataset_info["best_known_clique"]
        dataset_path = os.path.join(DATA_DIR, dataset_name)

        print("\n================================================")
        print(f"Dataset: {dataset_name}")
        print("================================================")

        A = load_mtx_graph(dataset_path)

        for algorithm_info in ALGORITHMS:
            algorithm_name = algorithm_info["name"]
            algorithm_function = algorithm_info["function"]

            print(f"\nRunning {algorithm_name}...")

            start_time = time.time()

            summary = algorithm_function(
                A,
                num_starts=NUM_STARTS,
                max_iter=MAX_ITER,
                tol=TOL,
                seed=SEED,
                start_mode=START_MODE,
            )

            runtime = time.time() - start_time
            best = summary["best_result"]

            best_clique_size = summary["best_clique_size"]
            gap_to_best_known = best_known - best_clique_size
            ratio_to_best_known = best_clique_size / best_known

            row = {
                "dataset": dataset_name,
                "algorithm": algorithm_name,
                "best_known_clique": best_known,
                "best_found_clique": best_clique_size,
                "gap_to_best_known": gap_to_best_known,
                "ratio_to_best_known": ratio_to_best_known,
                "best_objective": summary["best_objective"],
                "best_result_is_valid_clique": summary["best_result_is_valid_clique"],
                "success_rate_clique_output": summary["success_rate_clique_output"],
                "num_clique_outputs": summary["num_clique_outputs"],
                "num_starts": summary["num_starts"],
                "start_mode": summary["start_mode"],
                "mean_runtime_internal": summary["mean_runtime"],
                "total_runtime_internal": summary["total_runtime"],
                "wall_runtime": runtime,
                "best_start_id": best["start_id"],
                "best_start_type": best["start_type"],
                "best_iterations": best["iterations"],
                "best_final_gap": best["final_gap"],
                "best_final_clique_density": best["final_clique_density"],
                "best_vertices": list(map(int, best["selected_vertices"])),
            }

            # Algorithm-specific statistics.
            if algorithm_name == "Classical FW":
                row["mean_clique_size"] = summary["mean_valid_clique_size"]
                row["std_clique_size"] = summary["std_valid_clique_size"]
                row["mean_support_size_all_runs"] = summary["mean_support_size_all_runs"]
                row["std_support_size_all_runs"] = summary["std_support_size_all_runs"]
            else:
                row["mean_clique_size"] = summary["mean_clique_size"]
                row["std_clique_size"] = summary["std_clique_size"]
                row["mean_support_size_all_runs"] = summary["mean_clique_size"]
                row["std_support_size_all_runs"] = summary["std_clique_size"]

            row["mean_objective"] = summary["mean_objective"]
            row["std_objective"] = summary["std_objective"]

            all_rows.append(row)

            print(f"Best found clique: {best_clique_size}")
            print(f"Best known clique: {best_known}")
            print(f"Gap to best known: {gap_to_best_known}")
            print(f"Valid clique success rate: {summary['success_rate_clique_output']}")
            print(f"Runtime: {runtime:.4f} seconds")

    save_results_csv(all_rows, RESULTS_CSV)

    total_runtime = time.time() - total_start_time

    print("\n================================================")
    print("Experiment finished")
    print("================================================")
    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Total wall runtime: {total_runtime:.4f} seconds")


def save_results_csv(rows, output_path):
    """
    Save experiment results to CSV.
    """
    if len(rows) == 0:
        raise ValueError("No results to save.")

    fieldnames = [
        "dataset",
        "algorithm",
        "best_known_clique",
        "best_found_clique",
        "gap_to_best_known",
        "ratio_to_best_known",
        "best_objective",
        "best_result_is_valid_clique",
        "success_rate_clique_output",
        "num_clique_outputs",
        "num_starts",
        "start_mode",
        "mean_clique_size",
        "std_clique_size",
        "mean_support_size_all_runs",
        "std_support_size_all_runs",
        "mean_objective",
        "std_objective",
        "mean_runtime_internal",
        "total_runtime_internal",
        "wall_runtime",
        "best_start_id",
        "best_start_type",
        "best_iterations",
        "best_final_gap",
        "best_final_clique_density",
        "best_vertices",
    ]

    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    run_experiment()