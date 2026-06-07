"""
plot_results.py

Generate plots from results/results_summary.csv.

Run from repository root:

    python3 code/plot_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "results_summary.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")


def save_bar_plot_best_clique(df):
    pivot = df.pivot(index="dataset", columns="algorithm", values="best_found_clique")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Best Clique Size Found by Each Algorithm")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best clique size")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "best_clique_sizes.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_bar_plot_mean_clique(df):
    pivot = df.pivot(index="dataset", columns="algorithm", values="mean_clique_size")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Mean Clique Size Across All Starts")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean clique size")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "mean_clique_sizes.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_bar_plot_ratio_to_best_known(df):
    pivot = df.pivot(index="dataset", columns="algorithm", values="ratio_to_best_known")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Ratio to Best-Known Clique Size")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best found / best known")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.1)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "ratio_to_best_known.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_bar_plot_success_rate(df):
    pivot = df.pivot(index="dataset", columns="algorithm", values="success_rate_clique_output")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Valid Clique Output Success Rate")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Success rate")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.1)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "clique_success_rate.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_bar_plot_runtime(df):
    pivot = df.pivot(index="dataset", columns="algorithm", values="total_runtime_internal")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Total Runtime by Algorithm")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Runtime in seconds")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "runtime_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_summary_table_plot(df):
    table_df = df[
        [
            "dataset",
            "algorithm",
            "best_known_clique",
            "best_found_clique",
            "gap_to_best_known",
            "success_rate_clique_output",
        ]
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "results_table.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"Could not find {RESULTS_CSV}. Run python3 code/main.py first."
        )

    df = pd.read_csv(RESULTS_CSV)

    save_bar_plot_best_clique(df)
    save_bar_plot_mean_clique(df)
    save_bar_plot_ratio_to_best_known(df)
    save_bar_plot_success_rate(df)
    save_bar_plot_runtime(df)
    save_summary_table_plot(df)

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
