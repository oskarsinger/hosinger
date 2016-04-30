import numpy as np

from numpy.linalg import svd

from linal.random import get_orthonormal_basis as get_ob

def get_svd(A, k=None, q=2):

    (m, n) = A.shape
    max_rank = min([m,n])

    if k is None:
        k = max_rank
    elif k > max_rank:
        raise ValueError(
            'The value of k cannot exceed the smallest dimension of A.')

    Q = get_ob(A, k, q)
    B = np.dot(Q.T, A)
    U, s, V = svd(B)
    U = np.dot(Q, U)

    return (U, np.diag(s), V)
