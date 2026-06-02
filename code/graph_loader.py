import os
import networkx as nx
from scipy.io import mmread
import numpy as np

def load_mtx_graph(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The archive {filepath} is not found in the directory.")

    # mmread reads the .mtx file and converts it to a sparse matrix
    sparse_matrix = mmread(filepath)
    
    # Converts the sparse matrix to a NetworkX graph
    G = nx.from_scipy_sparse_array(sparse_matrix)
    
    # Cleanup: ensure there are no self-loops (diagonals in A)
    # The model assumes a_ii = 0 for all i
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Returns the adjacency matrix in NumPy Array format
    A = nx.to_numpy_array(G)
    
    # Binarize the matrix (unweighted)
    A = (A > 0).astype(float)
    
    # Double check: no self-loops
    np.fill_diagonal(A, 0.0)
    
    # Check that the matrix is symmetric
    A = np.maximum(A, A.T)
    
    return A