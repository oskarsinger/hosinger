import numpy as np

from drrobert.misc import prod

def rademacher(size=None, p=None):

    choices = np.array([-1, +1])

    return np.random.choice(choices, size=size, p=p)

def normal(loc=0.0, scale=1.0, shape=(1)):

    if len(shape) > 1:
        size = prod(shape)

    vec = np.random.normal(
        loc=loc, scale=scale, size=size)

    return np.reshape(vec, shape)
