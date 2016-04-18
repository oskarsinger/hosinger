import numpy as np

from numpy.random import randn, choice
from linal.utils import quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged as is_conv

def is_k_valid(ds_list, k):

    p = min([ds.cols() for ds in ds_list])

    return k < p

def is_converged(
    unn_Phi_pairs,
    epsilons, 
    verbose):

    conv_info = zip(unn_Phi_pairs, epsilons)

    return [is_conv(unn_Phi_t, unn_Phi_t1, eps, verbose)
            for (unn_Phi_t, unn_Phi_t1), eps in conv_info]

def get_2way_objective(X, Phi, Y, Psi):

    aux_Psi = np.dot(Y, Psi)

    return get_objective([X], [Phi], aux_Psi)

def get_objective(Xs, Phis, Psi):

    if not len(Xs) == len(Phis):
        raise ValueError(
            'Xs and Phis should have the same number of elements.')

    X_transforms = [np.dot(X, Phi)
                    for X, Phi in zip(Xs, Phis)]
    residuals = [np.linalg.norm(X_trans - Psi)
                 for X_trans in X_transforms]

    return sum(residuals)

def get_gradient(X, unnormed, Psi):

    transformed_X = np.dot(X, unnormed)
    diff = transformed_X - Psi

    return np.dot(X.T, diff) / X.shape[0]

def get_init_basis_pairs(Sxs, k):

    return [get_init_basis_pair(Sx, k)
            for Sx in Sxs]

def get_init_basis_pair(Sx, k):

    # Initialize Gaussian matrices for unnormalized bases
    unn_Phi = randn(Sx.shape[0], k)

    # Normalize for initial normalized bases
    Phi = get_gram_normed(unn_Phi, Sx)

    return (Phi, unn_Phi)

def get_gram_normed(unnormed, S):

    basis_quad = quad(unnormed, S)
    normalizer = get_svd_power(basis_quad, -0.5)

    return np.dot(unnormed, normalizer)
