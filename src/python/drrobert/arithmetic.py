import numpy as np

from math import log, ceil

def get_uni_quad_sols(a, b, c):

    (a, b, c) = [float(i) for i in [a,b,c]]
    term1 = -b
    term2 = (b**2 - 4 * a * c)**(0.5)
    denom = 2 * a
    sol_pos = (term1 + term2) / denom
    sol_neg = (term1 - term2) / denom

    return (sol_pos, sol_neg)

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

def get_running_avg(old, new, i):

    weighted_old = (i - 1) * old

    return (weighted_old + new) / i

def get_moving_avg(old, new, beta):

    weighted_old = beta * old
    weighted_new = (1 - beta) * new

    return weighted_old + weighted_new

def int_ceil_log(c):

    return int(ceil(log(c)))
