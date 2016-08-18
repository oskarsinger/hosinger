import numpy as np

from svd_funcs import get_svd_power
from utils.products import get_quadratic as gq

def get_woodbury_inversion(H, rho):

    (m, d) = H.shape
    I_m = np.identity(m)
    I_d = np.identity(d)
    to_invert = rho*I_m + np.dot(H, H.T)
    inversion = get_svd_power(to_invert, -1)
    quad = gq(H, inversion)
    unscaled = I_d - quad

    return rho**(-1) * unscaled
