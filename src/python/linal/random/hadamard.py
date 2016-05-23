import numpy as np

from scipy.linalg import hadamard

from drrobert.random import rademacher
from linal.random.utils import get_rand_I_rows as get_rir
from linal.utils import multi_dot
from linal.structured import get_normed_hadamard

def get_hadamard_sketching_matrix(n, r, p=None):

    if r >= n:
        raise ValueError(
            'Parameter r must be strictly less than parameter n.')

    H = get_normed_hadamard(n)
    D = np.diag(rademacher(size=n))
    R = get_rir(n, r, p=p)
    constant = (n * 1.0 / r)**(0.5)

    return constant * multi_dot([R,H,D])
