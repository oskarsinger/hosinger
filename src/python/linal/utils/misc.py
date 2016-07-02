import numpy as np

def get_largest_entries(s, energy=None, k=None):

    n = s.shape[0]

    if k is not None and energy is not None:
        raise ValueError(
            'At most one of the k and energy parameters should be set.')

    if k is not None:
        if k > n:
            raise ValueError(
                'The value of k must not exceed length of the input vector.')

    if energy is not None and (energy <= 0 or energy >= 1):
        raise ValueError(
            'The value of energy must be in the open interval (0,1).')

    s = np.copy(s)

    if not (s == np.zeros_like(s)).all():
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

def get_array_mod(a, divisor, axis=0):

    shape = a.shape
    length = shape[axis]
    remainder = length % divisor
    end = length - remainder
    r_value = a[:end] if len(shape) == 1 else a[:end,:]

    return r_value
