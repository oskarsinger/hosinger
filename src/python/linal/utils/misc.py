import numpy as np
import drrobert.debug as drdb

def get_non_nan(X):

    non_nan_indexes = np.logical_not(np.isnan(X))

    return X[non_nan_indexes]

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

def get_thresholded(x, upper=None, lower=None):

    new_x = np.copy(x)

    if upper is not None:
        upper = np.ones_like(x) * upper
        upper_idx = new_x > upper
        new_x[upper_idx] = upper[upper_idx]

    if lower is not None:
        lower = np.ones_like(x) * lower

        exception_msg = ' '.join([
            'x:', str(x), 
            '\n',
            'lower:', str(lower)])

        def less():
            return new_x < lower

        lower_idx = drdb.handle_runtime_warning(
            less, exception_msg)
        new_x[lower_idx] = lower[lower_idx]

    return new_x

def get_safe_power(s, power):

    new = np.zeros_like(s)
    masked_s = np.ma.masked_invalid(s)

    if power < 0:
        non_zero = masked_s != 0
        new[non_zero] = np.power(
            masked_s[non_zero], power)
    else:
        new = np.power(masked_s, power)

    return new.filled(0)

def get_array_mod(a, divisor, axis=0):

    length = a.shape[axis]
    remainder = length % divisor
    end = length - remainder
    truncd = None
    
    if axis == 0:
        truncd = a[:end] if a.ndim == 1 else a[:end,:]
    else:
        truncd = a[:,:end]

    return truncd
