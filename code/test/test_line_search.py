import numpy as np

from graph_loader import load_mtx_graph
from objective import simplex_uniform_point, minimization_objective, gradient
from lmo import simplex_lmo
from line_search import exact_line_search


A = load_mtx_graph("Data/C125-9.mtx")
n = A.shape[0]

x = simplex_uniform_point(n)

g = gradient(A, x)
s, i = simplex_lmo(g)

d = s - x
gamma = exact_line_search(A, x, d, gamma_max=1.0)

f_before = minimization_objective(A, x)
f_after = minimization_objective(A, x + gamma * d)

print("Selected FW index:", i)
print("Gamma:", gamma)
print("f before:", f_before)
print("f after:", f_after)
print("Improved:", f_after <= f_before)