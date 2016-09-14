import numpy as np

from math import log, ceil

def get_running_variance(
    old_var, new, i, old_avg=None):

    avg = None

    if old_avg is not None:
        avg = get_running_avg(old_avg, new, i)
    else:
        old_avg = np.zeros_like(new)
        avg = np.zeros_like(new)

    if i > 2:
        old_var = (i - 2) * old_var

    new_var = old_var + (new - old_avg) * (new - avg)

    if i > 1:
        new_var /= (i - 1)

    return new_var

def get_running_variance(
    old_var, new, alpha, beta, 
    old_avg=None, 
    alpha_avg=None, 
    beta_avg=None):

    avg = None

    if old_avg is not None:
        avg = get_moving_avg(
            old_avg, new, alpha_avg, beta_avg)
    else:
        old_avg = np.zeros_like(new)
        avg = np.zeros_like(new)

    #TODO: finish this

def get_running_avg(old, new, i):

    alpha = 1.0 / i
    beta = 1.0
    new = new - old

    return get_moving_avg(
        old, new, alpha, beta)

def get_moving_avg(old, new, alpha, beta):

    weighted_old = beta * old
    weighted_new = alpha * new

    return weighted_old + weighted_new

def int_ceil_log(c):

    return int(ceil(log(c)))
