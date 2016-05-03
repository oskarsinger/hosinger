import numpy as np

def rademacher(size=None, p=None):

    choices = np.array([-1, +1])

    return np.random.choice(choices, size=size, p=p)
