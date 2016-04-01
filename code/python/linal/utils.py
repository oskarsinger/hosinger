import numpy as np

def multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_rank_k(m, n, k):

    if k > m:
        raise ValueError(
            'The value of k must not exceed the matrix dimension.')

    A = np.zeros(m,n)

    for i in range(m):
        u = np.random.randn(m,1)
        v = np.random.randn(1,n)
        A = A + np.dot(u, v)

    return A

def get_t_regged_gram(A, reg_const):

    gram = np.dot(A.T, A)
    reg_matrix = reg * np.identity(A.shape[1])

    return (gram + reg_matrix) / A.shape[0]
