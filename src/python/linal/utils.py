import numpy as np

def multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_rank_k(m, n, k):

    if k > min([m, n]):
        raise ValueError(
            'The value of k must not exceed the minimum matrix dimension.')

    A = np.zeros(m,n)

    for i in range(m):
        u = np.random.randn(m,1)
        v = np.random.randn(1,n)
        A = A + np.dot(u, v)

    return A

def weighted_sum_of_op(weights, matrix):

    (n, p) = matrix.shape

    if not n == len(weights):
        raise ValueError(
            'Number of rows in matrix must equal number of weights.')

    total = np.zeros((p,p))

    for i, weight in enumerate(weights):

        row = matrix[i,:]
        total += weight * np.dot(row, row.T)

    return total

def get_lms(weights, matrices):

    if not len(weights) == len(matrices):
        raise ValueError(
            'Number of weights must equal number of matrices.')

    (n, p) = matrices[0].shape
    total = np.zeros(matrices[0].shape)

    for weight, matrix in zip(weights, matrices):

        (n1, p1) = matrix.shape

        if not (n1 == n and p1 == p):
            raise ValueError(
                'Input matrices must all have same shape.')

        total += weight * matrix

    return total

def matrix_ip(A, B):

    mp = np.dot(A, B)

    return np.trace(mp)
