import numpy as np

import time

from numpy.linalg import qr, svd

def get_svd_r(A, k=None, q=2):

    (m, n) = A.shape
    max_rank = min([m,n])

    if k is None:
        k = max_rank
    elif k > max_rank:
        raise ValueError(
            'The value of k cannot exceed the smallest dimension of A.')

    before = time.clock()
    Q = get_orthonormal_basis_r(A, k, q)
    after = time.clock()
    print 'get_orthonormal_basis_r took', after-before, 'seconds.'

    before = time.clock()
    B = np.dot(Q.T, A)
    after = time.clock()
    print 'The matrix product of Q.T and A took', after-before, 'seconds.'

    before = time.clock()
    U, s, V = svd(B)
    after = time.clock()
    print 'The svd of B took', after-before, 'seconds.'

    before = time.clock()
    U = np.dot(Q, U)
    after = time.clock()
    print 'The matrix product of Q and U took', after-before, 'seconds.'

    return (U, np.diag(s), V)

def get_orthonormal_basis_r(A, k, q=2):

    # Careful here about k and l!
    (m, n) = A.shape
    max_rank = min([m,n])

    if k > max_rank:
        raise ValueError(
            'The value of k cannot exceed the smallest dimension of A.')

    Omega = np.random.randn(n, k)

    before = time.clock()
    Y = np.dot(A, Omega)
    after = time.clock()
    print 'The matrix product of A and Omega took', after-before, 'seconds.'

    before = time.clock()
    (Q, R) = qr(Y)
    after = time.clock()
    print 'The numpy QR decomp of Y took', after-before, 'seconds.'

    before = time.clock()
    for i in range(q):
        Y = np.dot(A.T, Q)
        (Q, R) = qr(Y)
        Y = np.dot(A, Q)
        (Q, R) = qr(Y)
    after = time.clock()
    print 'The power iteration on A and Y took', after-before, 'seconds.'

    return Q
