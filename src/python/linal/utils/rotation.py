import numpy as np

from linal.svd import get_svd_power
from linal.utils import get_multi_dot

def get_rotation(dim, angle, P, P_inv=None):

    if P_inv is None:
        P_inv = get_svd_power(P, -1)

    A = np.eye(dim)

    A[0,0] = np.cos(angle)
    A[1,1] = np.cos(angle)
    A[0,1] = -np.sin(angle)
    A[1,0] = np.sin(angle)

    return get_multi_dot(
        [P_inv, A, P])
