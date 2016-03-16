import numpy as np

def multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_svd_invert(A, power=-1, k=None, random=True):

    (U, s, V) = (None, None, None)

    if (not k is None) and (not random):
        raise ValueError(
            'The value of k can only be set if randomized SVD is used.')

    if random:
        (U, s, V) = get_svd_r(A, k=k)
    else:
        (U, s, V) = np.linalg.svd(A)

    power_vec = np.ones(s.shape)
    power_vec[s != 0] = power

    return multi_dot([U, np.diag(np.power(s, power_vec)), V])

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
