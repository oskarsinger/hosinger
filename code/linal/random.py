import numpy as np

from numpy.linalg import qr, svd

def get_svd_r(A, k=None, q=2, p=None):

    (m, n) = A.shape
    max_rank = min([m,n])

    if k is None:
        k = max_rank
    elif k > max_rank:
        raise ValueError(
            'The value of k cannot exceed the smallest dimension of A.')

    Q = get_orthonormal_basis_r(A, k, q)

    B = np.dot(Q.T, A)
    U, s, V = svd(B)
    U = np.dot(Q, U)

    return (U, np.diag(s), V)

def get_orthonormal_basis_r(A, k, q=2):

    # Careful here about k and l!
    (m, n) = A.shape
    max_rank = min([m,n])

    if l > max_rank:
        raise ValueError(
            'The value l cannot exceed the smallest dimension of A.')

    Omega = np.random.randn(n, l)
    Y = np.dot(A, Omega)
    (Q, R) = qr(Y)

    for i in range(q):
        Y = np.dot(A.T, Q)
        (Q, R) = qr(Y)
        Y = np.dot(A, Q)
        (Q, R) = qr(Y)

    return Q
