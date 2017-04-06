import numpy as np

from .products import get_multi_dot as gmd

def get_sherman_morrison(A_inv, b, c):
    # Sherman-Morrison update from Matrix Cookbook

    numerator = np.dot(
        np.dot(A_inv, b),
        np.dot(c.T, A_inv))
    denominator = 1 + gmd([c.T, A_inv, b])
    update = numerator / denominator

    return A_inv - update
