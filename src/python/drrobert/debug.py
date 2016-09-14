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

def check_for_large_numbers(
    variable,
    loc_string,
    var_name,
    raise_error=True,
    exponent=5):

    try:
        has_large_pos = np.any(
            variable > 10**(exponent))
        has_large_neg = np.any(
            variable < -10**(exponent))
        start = var_name + ' at ' + loc_string + ' has values '
        end = ' inside.'
        pos_error = 'greater than ' + str(10**(exponent))
        neg_error = 'less than ' + str(10**(exponent))
        mk_msg = lambda x: start + x + end
        msg = None

        if has_large_pos and has_large_neg:
            error = pos_error + ' and ' + neg_error
            msg = mk_msg(error)
        elif has_large_pos:
            msg = mk_msg(pos_error)
        elif has_large_neg:
            msg = mk_msg(neg_error)

        if msg is not None:
            if raise_error:
                raise ValueError(
                    msg + '\nVariable:\n' + str(variable))
            else:
                print msg
                print str(variable)
    except TypeError:
        pass

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
            if raise_error:
                raise ValueError(
                    msg + '\nVariable:\n' + str(variable))
            else:
                print msg
                print str(variable)
    except TypeError:
        pass
