import numpy as np

from utils import multi_dot, get_largest_entries, get_safe_power

def get_schatten_p_norm(A, p, energy=0.95, k=None):

    get_trans = lambda s: get_safe_power(np.absolute(s), p)
    s = get_transformed_sv(A, get_trans, energy=energy, k=k)
    
    return sum(s)

def get_transformed_sv(A, get_trans, energy=0.95, k=None):

    s = np.linalg.svd(A, compute_uv=False)
    s = get_largest_entries(s, energy=energy, k=k)

    return get_trans(s)

def get_svd_power(A, power, energy=0.95, k=None):

    get_trans = lambda s: get_safe_power(s, power)

    return get_transformed_svd(A, get_trans, energy=energy, k=k)

def get_transformed_svd(A, get_trans, energy=0.95, k=None):

    (U, s, V) = np.linalg.svd(A)
    s = get_largest_entries(s, energy=energy, k=k)
    transformed_s = get_trans(s)

    return _get_multiplied_svd(U, transformed_s, V)

def _get_multiplied_svd(U, s, V):

    (n, p) = (U.shape[0], V.shape[0])
    sigma = _get_sigma(n, p, s)

    return multi_dot([U, sigma, V])

def _get_sigma(n, p, s):

    sigma = np.zeros((n,p))

    for i in xrange(s.shape[0]):
        sigma[i,i] = s[i]

    return sigma
