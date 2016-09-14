import numpy as np

import warnings

def print_and_return(val):

    print val

    return val

def handle_runtime_warning(func, exception_msg):

    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        try:
            return func()
        except RuntimeWarning:
            raise Exception(exception_msg)

def check_for_nan_or_inf(
    variable, 
    loc_string, 
    var_name, 
    raise_error=True):

    try:
        has_nan = np.any(np.isnan(variable))
        has_inf = np.any(np.isinf(variable))
        start = var_name + ' at ' + loc_string + ' has '
        end = ' values inside.'
        mk_msg = lambda x: start + x + end
        msg = None

        if has_nan and has_inf:
            msg = mk_msg('NaN and inf')
        elif has_nan:
            msg = mk_msg('NaN')
        elif has_inf:
            msg = mk_msg('inf')
        
        if msg is not None:
            print msg
            print str(variable)

            if raise_error:
                raise ValueError(msg)
    except TypeError:
        pass

