import spancca

import numpy as np

from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from linal.utils.misc import get_safe_power

def get_pearson_matrix(X1, X2):

    (n, p1) = X1.shape
    p2 = X2.shape[1]

    # Get means, stds, and zero-mean vars
    mu1 = np.nanmean(X1, axis=0)
    mu2 = np.nanmean(X2, axis=0)
    std1 = np.nanstd(X1, axis=0)
    std2 = np.nanstd(X2, axis=0)
    zm_X1 = X1 - mu1
    zm_X2 = X2 - mu2

    # Get masked zero-mean vars
    masked_X1 = np.ma.masked_invalid(zm_X1)
    masked_X2 = np.ma.masked_invalid(zm_X2)

    numerator = np.dot(
        masked_X1.T, masked_X2).filled(0) / n
    denominator = np.dot(
            std1[:,np.newaxis], 
            std2[:,np.newaxis].T)
    inv_denom = get_safe_power(
        denominator, -1)

    return numerator * inv_denom

def get_cca_vecs(X1, X2, num_nonzero=None):

    (x_weights, y_weights) = [None] * 2

    if num_nonzero is None:
        cca = CCA(n_components=1)

        cca.fit(X1, X2)

        x_weights = cca.x_weights_
        y_weights = cca.y_weights_
    else:
        x_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        y_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        A = get_pearson_matrix(X1, X2)
        T = X1.shape[0]
        rank = 7
        (x_weights, y_weights) = spancca.cca(
            A,
            rank,
            T,
            x_project,
            y_project)

    return (
        x_weights,
        y_weights)

