import numpy as np

from utils import multi_dot, get_top_entries, get_safe_power

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

    return get_sv_transformed(A, get_trans, energy=energy, k=k)

def get_transformed_svd(A, get_trans, energy=0.95, k=None):

    (U, s, V) = np.linalg.svd(A)
    s = get_largest_entries(s, energy=energy, k=k)
    transformed_s = np.diag(get_trans(s))

    return multi_dot([U, transformed_s, V])
