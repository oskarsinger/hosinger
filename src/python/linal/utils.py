import numpy as np

def multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_rank_k(m, n, k):

    if k > min([m, n]):
        raise ValueError(
            'The value of k must not exceed the minimum matrix dimension.')

    A = np.zeros((m,n))

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

def get_largest_entries(s, energy=None, k=None):

    n = s.shape[0]

    if k is not None and energy is not None:
        raise ValueError(
            'At most one of the k and energy parameters should be set.')

    if k is not None:
        if k > n:
            raise ValueError(
                'The value of k must not exceed the input vector.')

    if energy is not None and (energy <= 0 or energy >= 1):
        raise ValueError(
            'The value of energy must be in the open interval (0,1).')

    s = np.copy(s)

    if k is not None:
        s[k+1:] = 0
    elif energy is not None:
        total = sum(s)
        current = 0
        count = 0
        
        for i in range(n):
            if current / total < energy:
                current = current + s[i]
                count = count + 1

        s[count+1:] = 0

    return s

def get_thresholded(x, upper=float('Inf'), lower=0):

    if np.isscalar(upper):
        upper = np.zeros_like(x) + upper

    if np.isscalar(lower):
        lower = np.zeros_like(x) + lower

    upper_idx = x > upper
    lower_idx = x < lower
    new_x = np.copy(x)
    new_x[upper_idx] = upper[upper_idx]
    new_x[lower_idx] = lower[lower_idx]

    return new_x

def get_safe_power(s, power):

    power_vec = np.ones(s.shape)

    if power == 0:
        power_vec = np.zeros(s.shape)
    else:
        power_vec[s != 0] = power
    
    return np.power(s, power_vec)
