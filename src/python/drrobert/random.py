import numpy as np
import numpy.random as npr

from drrobert.misc import prod

def rademacher(size=None, p=0.5):

    choices = [-1, 1]
    ps = [1-p, p]

    return npr.choice(choices, size=size, p=ps)

def normal(loc=0.0, scale=1.0, shape=1):

    size = shape

    if type(shape) is tuple:
        size = prod(shape)
    elif type(shape) is not int:
        raise TypeError(
            'Parameter shape must of type tuple or int.')

    if scale == 0:
        vec = np.zeros(shape)
    else:
        vec = npr.normal(
            loc=loc, scale=scale, size=size)

        if type(shape) is tuple:
            vec = np.reshape(vec, shape)

    return vec

def log_uniform(upper, lower, size=1):

    log_u = np.log(upper)
    log_l = np.log(lower)
    logs = npr.uniform(
        low=log_l, high=log_u, size=size)

    if size == 1:
        logs = logs[0]

    return np.exp(logs)
