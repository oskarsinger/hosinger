import numpy as np

from .products import get_multi_dot as gmd
from .products import get_quadratic as gq
from linal.svd import get_svd_power

def get_woodbury_inversion(H, rho):

    (m, d) = H.shape
    I_m = np.identity(m)
    I_d = np.identity(d)
    to_invert = rho*I_m + np.dot(H, H.T)
    inversion = get_svd_power(to_invert, -1)
    quad = gq(H, inversion)
    unscaled = I_d - quad

    return rho**(-1) * unscaled

def get_sherman_morrison(A_inv, b, c):
    # Sherman-Morrison update from Matrix Cookbook

    numerator = np.dot(
        np.dot(A_inv, b),
        np.dot(c.T, A_inv))
    denominator = 1 + gmd([c.T, A_inv, b])
    update = numerator / denominator

    return A_inv - update
