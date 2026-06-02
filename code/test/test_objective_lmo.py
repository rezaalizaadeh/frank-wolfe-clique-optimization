import numpy as np

from graph_loader import load_mtx_graph
from objective import (
    max_clique_objective,
    minimization_objective,
    gradient,
    is_in_simplex,
    simplex_uniform_point,
    support_size,
)
from lmo import simplex_lmo, frank_wolfe_gap


A = load_mtx_graph("Data/C125-9.mtx")
n = A.shape[0]

x = simplex_uniform_point(n)

F = max_clique_objective(A, x, check_input=True)
f = minimization_objective(A, x)
g = gradient(A, x)

s, i = simplex_lmo(g)
gap, s_gap, i_gap = frank_wolfe_gap(A, x)

print("A shape:", A.shape)
print("x in simplex:", is_in_simplex(x))
print("support size:", support_size(x))
print("F(x) maximization objective:", F)
print("f(x) minimization objective:", f)
print("Gradient shape:", g.shape)
print("LMO index:", i)
print("LMO atom sum:", s.sum())
print("FW gap:", gap)
print("FW gap index:", i_gap)
