import numpy as np

from drrobert.misc import prod

def rademacher(size=None, p=None):

    choices = np.array([-1, +1])

    return np.random.choice(choices, size=size, p=p)

def normal(loc=0.0, scale=1.0, shape=1):

    size = shape

    if type(shape) is tuple:
        size = prod(shape)
    elif type(shape) is not int:
        raise TypeError(
            'Parameter shape must of type tuple or int.')

    vec = np.random.normal(
        loc=loc, scale=scale, size=size)

    if type(shape) is tuple:
        vec = np.reshape(vec, shape)

    return vec
