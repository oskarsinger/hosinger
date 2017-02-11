import spancca

import numpy as np

from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from linal.utils.misc import get_safe_power
from linal.svd_funcs import get_svd_power

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
        masked_X1.H, masked_X2).filled(0) / n
    denominator = np.dot(
            std1[:,np.newaxis], 
            std2[:,np.newaxis].H)
    inv_denom = get_safe_power(
        denominator, -1)

    return numerator * inv_denom

def get_cca_vecs(X1, X2, n_components=1, num_nonzero=None):

    (n1, p1) = X1.shape
    (n2, p2) = X2.shape
    n = None

    if not n1 == n2:
        raise ValueError(
            'X1 and X2 must have same 1st dimension.')
    else:
        n = n1

    (x1_weights, x2_weights) = [None] * 2

    if num_nonzero is None:
        # Get means and zero-mean vars
        mu1 = np.nanmean(X1, axis=0)
        mu2 = np.nanmean(X2, axis=0)
        zm_X1 = X1 - mu1
        zm_X2 = X2 - mu2

        # Get sample covariance matrices
        CX1 = np.dot(zm_X1.H, zm_X1) / n
        CX2 = np.dot(zm_X2.H, zm_X2) / n
        CX12 = np.dot(zm_X1.H, zm_X2) / n

        # Get inverse sqrts and normed sample cross-covariance
        CX1_inv_sqrt = get_svd_power(CX1, -0.5)
        CX2_inv_sqrt = get_svd_power(CX2, -0.5)
        Omega = np.dot(
            CX1_inv_sqrt,
            np.dot(CX12, CX2_inv_sqrt))

        # Get canonical vectors
        (U, s, V) = np.linalg.svd(Omega)
        (x1_weights, x2_weights) = (
            np.dot(CX1_inv_sqrt, U[:,0]), 
            np.dot(CX2_inv_sqrt, V.H[:,0]))
    else:
        x_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        y_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        A = get_pearson_matrix(X1, X2)
        T = X1.shape[0]
        rank = num_nonzero * 2 + 1
        (x_weights, y_weights) = spancca.cca(
            A,
            rank,
            T,
            x_project,
            y_project,
            verbose=False)

    return (
        x_weights,
        y_weights)
