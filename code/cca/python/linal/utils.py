import numpy as np

def multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def norm(X, A=None, order=None):

    if A is not None:
        norm = multi_dot([X.T, A, X])
    elif order is None:
        norm = np.linalg.norm(X)
    else:
        norm = np.linalg.norm(X, ord=order)

    return norm

def get_svd_invert(A, power=-1, k=None, random=True):

    (U, s, V) = (None, None, None)

    if (not k is None) and (not random):
        raise ValueError(
            'The value of k can only be set if randomized SVD is used.')

    if random:
        (U, s, V) = get_svd_r(A, k=k)
    else:
        (U, s, V) = np.linalg.svd(A)

    return multi_dot([U, np.diag(np.power(s, power)), V])

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
