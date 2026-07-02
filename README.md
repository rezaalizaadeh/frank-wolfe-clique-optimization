Frank-Wolfe Maximum Clique Optimization

Course project for Optimization Methods for Clustering.

This repository implements and compares Frank-Wolfe variants for the L2-regularized continuous formulation of the maximum clique problem on DIMACS graph benchmark instances.

⸻

Project Overview

The maximum clique problem asks for the largest subset of vertices in a graph such that every pair of vertices in the subset is connected by an edge.

In this project, a graph is represented by its adjacency matrix A, and the maximum clique problem is studied through the following continuous L2-regularized formulation:

$$
\max_{x \in \Delta} F(x) = x^T A x + \frac{1}{2}|x|_2^2
$$

where:

* A is the adjacency matrix of the graph;
* x is a vector of weights over the vertices;
* Delta is the probability simplex.

The probability simplex is defined as:

$$
\Delta = {x \in \mathbb{R}^n : x_i \geq 0,\ \sum_i x_i = 1}.
$$

Since the implemented Frank-Wolfe algorithms are written in minimization form, the code solves the equivalent problem:

$$
\min_{x \in \Delta} f(x) = -x^T A x - \frac{1}{2}|x|_2^2.
$$

The algorithms return continuous vectors over the simplex. Therefore, the final discrete clique is obtained using support thresholding followed by greedy clique extraction and repair.

⸻

Implemented Algorithms

The project implements and compares three Frank-Wolfe variants:

* Classical Frank-Wolfe
* Away-Step Frank-Wolfe
* Pairwise Frank-Wolfe

All methods use:

* Linear minimization oracle over the simplex
* Exact line search
* Multiple starting points
* Support thresholding
* Greedy clique extraction and repair
* Final clique validation
* Comparison with known benchmark clique sizes

⸻

Repository Structure

code/
├── graph_loader.py
├── objective.py
├── lmo.py
├── line_search.py
├── utils.py
├── frank_wolfe.py
├── away_step_fw.py
├── pairwise_fw.py
├── plot_results.py
└── main.py
data/
├── C125-9.mtx
├── brock200-2.mtx
└── keller4.mtx
results/
└── results_summary.csv
plots/
├── best_clique_sizes.png
├── ratio_to_best_known.png
├── mean_clique_sizes.png
├── runtime_comparison.png
├── clique_success_rate.png
└── results_table.png
report/
└── final_report.pdf

⸻

File Description

* graph_loader.py: loads Matrix Market .mtx graph files and converts them into cleaned adjacency matrices.
* objective.py: defines the maximum clique objective, minimization form, gradient, simplex utilities, and support utilities.
* lmo.py: implements the linear minimization oracle over the simplex, active-set oracle, and Frank-Wolfe gap.
* line_search.py: implements exact line search for the quadratic minimization objective.
* utils.py: contains helper functions for clique checking, clique density, clique extraction/repair, and random starting points.
* frank_wolfe.py: implements the Classical Frank-Wolfe method.
* away_step_fw.py: implements the Away-Step Frank-Wolfe method.
* pairwise_fw.py: implements the Pairwise Frank-Wolfe method.
* main.py: runs all algorithms on all datasets and saves the comparison table.
* plot_results.py: generates plots from the saved results.

⸻

Requirements

The code requires Python 3 and the following packages:

numpy
scipy
networkx
pandas
matplotlib

The dependencies can be installed with:

pip install numpy scipy networkx pandas matplotlib

⸻

How to Run

From the repository root, run:

python3 code/main.py

This runs all algorithms on all datasets and saves the summary table in:

results/results_summary.csv

To generate the plots, run:

python3 code/plot_results.py

The plots are saved in:

plots/

⸻

Experimental Setup

The final experiments use:

* 100 starting points
* uniform_random start mode
* Maximum iterations: 1000
* Frank-Wolfe gap tolerance: 1e-6
* Exact line search
* Support thresholding followed by greedy clique extraction/repair

The uniform_random start mode uses one uniform simplex start and the remaining starts as random simplex points.

The same experimental setup and post-processing procedure are used for all three methods to ensure a fair comparison.

⸻

Benchmark Datasets

The project uses three graph benchmark instances:

Dataset	Vertices	Edges	Density	Best Known Clique
C125-9	125	6963	0.8985	34
brock200-2	200	9876	0.4963	12
keller4	171	9435	0.6491	11

⸻

Results

The final best clique sizes found by the three methods are:

Dataset	Classical FW	Away-Step FW	Pairwise FW	Best Known
C125-9	34	33	33	34
brock200-2	10	10	10	12
keller4	11	11	11	11

Classical Frank-Wolfe reached the best known clique size on C125-9.

All three methods reached the best known clique size on keller4.

None of the methods reached the best known clique size on brock200-2, which was the most challenging instance among the selected benchmarks.

Overall, the three methods produced comparable clique-quality results after the common post-processing step. Away-Step Frank-Wolfe and Pairwise Frank-Wolfe were computationally faster in this implementation while maintaining competitive clique sizes.

⸻

Plots

The repository includes plots comparing the algorithms in terms of clique quality and runtime.

Best Clique Sizes

Ratio to Best-Known Clique Size

Mean Clique Sizes Across Starts

Runtime Comparison

Clique Success Rate

⸻

Notes

The maximum clique formulation used in this project is non-convex. Therefore, the Frank-Wolfe gap is used as a stationarity measure for the continuous optimization problem, not as a certificate of global optimality.

The algorithms return continuous simplex vectors. The reported clique sizes are obtained only after support thresholding and greedy clique extraction/repair.

The same post-processing procedure is applied consistently to all three Frank-Wolfe variants.

⸻

Authors

Course project by Group 11.

* Reza Alizadeh
* Miguel Cruz Real
