"""
main.py

Run Classical Frank-Wolfe, Away-Step Frank-Wolfe, and Pairwise Frank-Wolfe
on the selected maximum clique benchmark datasets.

Run from the repository root:

    python3 code/main.py

Expected folder structure:

    Project/
    ├── code/
    │   ├── main.py
    │   ├── graph_loader.py
    │   ├── objective.py
    │   ├── lmo.py
    │   ├── line_search.py
    │   ├── utils.py
    │   ├── frank_wolfe.py
    │   ├── away_step_fw.py
    │   └── pairwise_fw.py
    ├── data/
    │   ├── C125-9.mtx
    │   ├── brock200-2.mtx
    │   └── keller4.mtx
    └── results/
        └── results_summary.csv

All algorithms use the same experimental setup:
    - 100 starting points
    - uniform_random initialization
    - exact line search
    - support thresholding and greedy clique extraction after optimization
"""

import os
import csv
import time

from graph_loader import load_mtx_graph
from frank_wolfe import multistart_frank_wolfe
from away_step_fw import multistart_away_step_frank_wolfe
from pairwise_fw import multistart_pairwise_frank_wolfe


# ============================================================
# PROJECT PATHS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))       # .../Project/code
PROJECT_ROOT = os.path.dirname(HERE)                    # .../Project

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")


# ============================================================
# DATASETS AND ALGORITHMS
# ============================================================

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


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_STARTS = 100
MAX_ITER = 1000
TOL = 1e-6
SEED = 42
START_MODE = "uniform_random"


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    """
    Run all algorithms on all datasets and save a summary CSV.

    The reported clique is obtained after:
        1. optimizing the continuous objective,
        2. extracting the support using a threshold,
        3. applying greedy clique extraction if the support is not already a clique.

    This post-processing is applied consistently to all algorithms.
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

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}\n"
                f"Please make sure the .mtx files are inside the data/ folder."
            )

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
                "mean_clique_size": summary["mean_clique_size"],
                "std_clique_size": summary["std_clique_size"],
                "mean_objective": summary["mean_objective"],
                "std_objective": summary["std_objective"],
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

            # Optional diagnostic fields.
            # Some files report numerical/raw support size in addition to extracted clique size.
            if "mean_support_size_all_runs" in summary:
                row["mean_support_size_all_runs"] = summary["mean_support_size_all_runs"]
            else:
                row["mean_support_size_all_runs"] = summary["mean_clique_size"]

            if "std_support_size_all_runs" in summary:
                row["std_support_size_all_runs"] = summary["std_support_size_all_runs"]
            else:
                row["std_support_size_all_runs"] = summary["std_clique_size"]

            if "final_raw_support_size" in best:
                row["best_raw_support_size"] = best["final_raw_support_size"]
            else:
                row["best_raw_support_size"] = ""

            all_rows.append(row)

            print(f"Best found clique: {best_clique_size}")
            print(f"Best known clique: {best_known}")
            print(f"Gap to best known: {gap_to_best_known}")
            print(f"Ratio to best known: {ratio_to_best_known:.4f}")
            print(f"Best result is valid clique: {summary['best_result_is_valid_clique']}")
            print(f"Valid clique success rate: {summary['success_rate_clique_output']}")
            print(f"Runtime: {runtime:.4f} seconds")

    save_results_csv(all_rows, RESULTS_CSV)

    total_runtime = time.time() - total_start_time

    print("\n================================================")
    print("Experiment finished")
    print("================================================")
    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Total wall runtime: {total_runtime:.4f} seconds")


# ============================================================
# SAVE RESULTS
# ============================================================

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
        "best_raw_support_size",
        "best_vertices",
    ]

    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    run_experiment()
