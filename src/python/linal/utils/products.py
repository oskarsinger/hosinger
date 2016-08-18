import numpy as np

def get_mahalanobis_inner_product(A):

    inside_A = np.copy(A)

    def ip(x,y):

        return multi_dot([x.T, inside_A, y])

    return ip

def get_matrix_inner_product(A, B):

    mp = np.dot(A, B)

    return np.trace(mp)

def get_multi_dot(As):

    return reduce(lambda B,A: np.dot(B,A), As)

def get_quadratic(X, A):

    return multi_dot([X.T, A, X])

