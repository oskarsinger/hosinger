import numpy as np

from numpy.random import randn, choice
from linal.utils import quadratic as quad
from linal.mp_pinv import get_mp_pinv as get_pinv
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

    n = X.shape[0]
    transformed_X = np.dot(X, unnormed)
    diff = transformed_X - Psi

    return np.dot(X.T, diff) / n

def get_2way_basis_update(X1, X2, unnormed1, normed2, S1, eta1):

    Psi = np.dot(X2, normed2)

    return get_basis_update(X1, unnormed1, Psi, S1, eta1)

def get_basis_update(X, unnormed, Psi, Sx, eta1):

    # Calculate gradient for 2-way CCA
    gradient = get_gradient(X, unnormed, Psi)

    # Take a gradient step on unnormed1
    unnormed_next = unnormed - eta1 * gradient

    # Normalize unnormed 1 with inversion of matrix quadratic
    normed = get_gram_normed(unnormed_next, Sx)

    return (unnormed_next, normed)

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
    normalizer = get_pinv(
        basis_quad, energy=0.95, sqrt=True)

    return np.dot(unnormed, normalizer)
