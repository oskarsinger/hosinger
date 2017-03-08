import numpy as np

from numpy.linalg import qr

def get_orthonormal_basis(A, k, q=2):

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
    print( 'The matrix product of A and Omega took', after-before, 'seconds.' )

    before = time.clock()
    (Q, R) = qr(Y)
    after = time.clock()
    print( 'The numpy QR decomp of Y took', after-before, 'seconds.' )

    before = time.clock()
    for i in range(q):
        Y = np.dot(A.T, Q)
        (Q, R) = qr(Y)
        Y = np.dot(A, Q)
        (Q, R) = qr(Y)
    after = time.clock()
    print( 'The power iteration on A and Y took', after-before, 'seconds.' )

    return Q
