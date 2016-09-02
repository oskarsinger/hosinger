from math import log, ceil

def get_running_avg(old, new, i):

    alpha = 1.0 / i
    beta = float(i-1) / i

    return get_moving_avg(
        old, new, alpha, beta)

def get_moving_avg(old, new, alpha, beta):

    weighted_old = self.beta * old
    weighted_new = self.alpha * new

    return weighted_old + weighted_new

def int_ceil_log(c):

    return int(ceil(log(c)))
