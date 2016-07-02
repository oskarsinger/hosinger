import numpy as np

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
