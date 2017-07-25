import spancca

import numpy as np

from scipy.stats import pearsonr
from linal.utils import get_safe_power
from linal.utils import get_quadratic, get_multi_dot
from linal.svd import get_svd_power

def get_zm_uv(X):

    (n, p) = X.shape
    mu = np.nanmean(X, axis=0)
    zm_X = X - mu

    return zm_X / n**(0.5)

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

def get_cca_vecs(X1, X2, n_components=1, num_nonzero=None):

    (n1, p1) = X1.shape
    (n2, p2) = X2.shape
    n = None

    if not n1 == n2:
        raise ValueError(
            'X1 and X2 must have same 1st dimension.')
    else:
        n = n1

    # Get means, zero-mean, normed vars
    mu1 = np.nanmean(X1, axis=0)
    mu2 = np.nanmean(X2, axis=0)
    normalizer = n**(0.5)
    zm_X1 = (X1 - mu1) / normalizer
    zm_X2 = (X2 - mu2) / normalizer

    # Get sample covariance matrices
    CX1 = np.dot(zm_X1.T, zm_X1)
    CX2 = np.dot(zm_X2.T, zm_X2)
    CX12 = np.dot(zm_X1.T, zm_X2)

    # Get inverse sqrts and normed sample cross-covariance
    CX1_inv_sqrt = get_svd_power(CX1, -0.5)
    CX2_inv_sqrt = get_svd_power(CX2, -0.5)

    Omega = get_multi_dot([
        CX1_inv_sqrt,
        CX12, 
        CX2_inv_sqrt])
    (Phi1, Phi2) = [None] * 2

    if num_nonzero is None:
        (U, s, V) = np.linalg.svd(Omega)
        unnormed_Phi1 = U[:,:n_components]
        unnormed_Phi2 = V.T[:,:n_components]
        Phi1 = np.dot(CX1_inv_sqrt, unnormed_Phi1)
        Phi2 = np.dot(CX2_inv_sqrt, unnormed_Phi2)
    else:
        x_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        y_project = spancca.projections.setup_sparse(
            nnz=num_nonzero)
        rank = min([
            num_nonzero * 2 + 1,
            min(Omega.shape) - 1])
        (Phi1, Phi2) = spancca.cca(
            Omega,
            rank,
            n,
            x_project,
            y_project,
            verbose=False)

    projected1 = np.dot(zm_X1, Phi1)
    projected2 = np.dot(zm_X2, Phi2)
    cc = np.sum(
        projected1 * projected2, 
        axis=1)[:,np.newaxis]

    return (Phi1, Phi2, cc)
