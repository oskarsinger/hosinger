import numpy as np

def is_fully_connected(G):

    L = get_Laplacian(G)
    L_rank = np.linalg.matrix_rank(L)

    return L_rank == L.shape[0]

def get_Laplacian(G):

    d = np.sum(G, axis=1)
    
    return np.diag(d) - G
