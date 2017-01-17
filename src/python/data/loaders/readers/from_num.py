import numpy as np

from linal.utils import get_safe_power

def get_row_magnitude(a):

    fac = get_fields_as_columns(a)
    squares = get_safe_power(fac, 2)
    sums = np.sum(squares, axis=1)

    return get_safe_power(sums, 0.5)[:,np.newaxis]

def get_array_as_is(a):

    indexed = None

    if len(a.shape) == 2:
        indexed = a[:,:]
    else:
        indexed = a[:]

    return indexed

def get_fields_as_columns(a):

    fields = [a[name][:,np.newaxis] 
              for name in a.dtype.names]

    return np.hstack(fields)
