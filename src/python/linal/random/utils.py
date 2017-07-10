import numpy as np

def get_sparse_normal(m, n, density=0.1):

    A = np.random.randn(m, n)
    sample = np.random.uniform(size=(m,n))
    fill_spots = (sample < density).astype(int)
    
    return A * fill_spots

def get_rand_I_rows(n, r, p=None):

    rows = np.choice(
        range(n), size=r, replace=False, p=p)

    return np.identity(n)[rows,:]

def get_rank_k(m, n, k):

    if k > min([m, n]):
        raise ValueError(
            'The value of k must not exceed the minimum matrix dimension.')

    A = np.zeros((m,n))

    U = np.random.randn(m, k)
    V = np.random.randn(k, n)

    return np.dot(U, V)

