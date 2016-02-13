def multi_dot(As):

    B = np.identity(As[0].shape[0])

    for A in As:
        B = np.dot(B, A)

    return B

def quadratic(X, A):

    return multi_dot([X.T, A, X])

def get_svd_invert(A, k=None, random=True):

    (U, s, V) = (None, None, None)

    if (not k is None) and (not random):
        raise ValueError(
            'The value of k can only be set if randomized SVD is used.')

    if random:
        (U, s, V) = get_svd_r(A, k=k)
    else:
        (U, s, V) = np.linalg.svd(A)

    return multi_dot([U, np.power(s, -0.5), V])
