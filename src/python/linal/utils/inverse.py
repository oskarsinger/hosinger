import numpy as np

from linal.utils import get_multi_dot as gmd

def get_rank1_inv_update(A_inv, b, c):
    # Sherman-Morrison update from Matrix Cookbook

    numerator = np.dot(
        np.dot(A_inv, b),
        np.dot(c.T, A_inv))
    denominator = 1 + gmd([c.T, A_inv, b])
    update = numerator / denominator

    return A_inv - update
