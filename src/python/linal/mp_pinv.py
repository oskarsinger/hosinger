import numpy as np

from utils import multi_dot, get_top_entries

def get_mp_pinv(A, energy=None, k=None, sqrt=False, random=False):

    (n, p) = A.shape
    power = -0.5 if sqrt else -1
    (U, s, V) = (None, None, None)

    if random:
        (U, s, V) = get_svd_r(A, k=k, energy=energy)
    else:
        (U, s, V) = np.linalg.svd(A)

        s = get_top_entries(s, energy=energy, k=k)

    pseudo_inv_sigma = np.diag(_get_safe_power(s, power))

    return multi_dot([U, pseudo_inv_sigma, V])

def _get_safe_power(s, power):

    power_vec = np.ones(s.shape)

    if power == 0:
        power_vec = np.zeros(s.shape)
    else:
        power_vec[s != 0] = power
    
    return np.power(s, power_vec)
