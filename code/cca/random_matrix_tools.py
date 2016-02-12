import numpy as np

from numpy.linalg import qr, svd

def get_svd_r(A, k, q=2, p=None):

    if p is None:
        p = math.log(k)

    (m, n) = A.shape
    Q = get_orthonormal_basis(Omega, k+p, q) #Figure out what to replace those Nones with

    B = np.dot(Q.T, A)
    U, s, V = svd(B)
    U = np.dot(Q, U)

    return (U, s, V)

def get_orthonormal_basis_r(A, l, q=2):

    (m, n) = A.shape
    Omega = np.randn(n, l)
    Y = np.dot(A, Omega)
    (Q, R) = qr(Y)

    for i in range(q):
        Y = np.dot(A.T, Q)
        (Q, R) = qr(Y)
        Y = np.dot(A, Q)
        (Q, R) = qr(Y)

    return Q
