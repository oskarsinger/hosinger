import numpy as np

from utils import multi_dot

def get_mp_pinv(A, energy=None, k=None, sqrt=False, random=False):

    (n, p) = A.shape
    power = -0.5 if sqrt else -1
    (U, s, V) = (None, None, None)

    if random:
        (U, s, V) = get_svd_r(A, k=k, energy=energy)
    else:
        (U, s, V) = np.linalg.svd(A)

        s = _get_thresholded(s, energy=energy, k=k)

    print s

    pseudo_inv_sigma = np.diag(_get_safe_power(s, power))

    return multi_dot([U, pseudo_inv_sigma, V])

def _get_safe_power(s, power):

    power_vec = np.ones(s.shape)

    if power == 0:
        power_vec = np.zeros(s.shape)
    else:
        power_vec[s != 0] = power
    
    return np.power(s, power_vec)

def _get_thresholded(s, energy=None, k=None):

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
