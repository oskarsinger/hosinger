import numpy as np

def print_and_return(val):

    print val

    return val

def check_for_nan_or_inf(
    np_array, 
    loc_string, 
    var_name, 
    raise_error=False):

    has_nan = np.any(np.isnan(np_array))
    has_inf = np.any(np.isinf(np_array))
    start = var_name + ' at ' + loc_string + ' has '
    end = ' values inside.'
    msg = None

    if has_nan and has_inf:
        msg = start + 'NaN and inf' + end
    elif has_nan:
        msg = start + 'NaN' + end
    elif has_inf:
        msg = start + 'inf' + end
    
    if msg is not None:
        print msg

        if raise_error:
            raise ValueError(msg)
