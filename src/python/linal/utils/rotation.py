import numpy as np

from linal.svd import get_svd_power
from linal.utils import get_multidot

def get_rotation(dim, angle, P=None):

    if P is none:
        P = np.eye(dim)

    P_inv = get_svd_power(P, -1)
    A = np.eye(n)

    A[0,0] = np.cos(angle)
    A[1,1] = np.cos(angle)
    A[0,1] = -np.sin(angle)
    A[1,0] = np.sin(angle)

    return get_multidot(
        [P_inv, A, P])
