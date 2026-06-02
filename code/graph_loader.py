import os
import networkx as nx
from scipy.io import mmread

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
    return nx.to_numpy_array(G)