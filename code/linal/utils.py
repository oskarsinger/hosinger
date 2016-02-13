import numpy as np

def multi_dot(As):

    B = np.identity(As[0].shape[0])

    for A in As:
        B = np.dot(B, A)

    return B

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_svd_invert(A, k=None, random=True):

    (U, s, V) = (None, None, None)

    if (not k is None) and (not random):
        raise ValueError(
            'The value of k can only be set if randomized SVD is used.')

    if random:
        (U, s, V) = get_svd_r(A, k=k)
    else:
        (U, s, V) = np.linalg.svd(A)

    return multi_dot([U, np.power(s, -0.5), V])

def get_square_rank_k(m, k):

    if k > m:
        raise ValueError(
            'The value of k must not exceed the matrix dimension.')

    A = np.zeros(m,n)

    for i in range(m):
        v = np.random.randn(m,1)
        A = A + np.dot(v, v.T)

    return A
