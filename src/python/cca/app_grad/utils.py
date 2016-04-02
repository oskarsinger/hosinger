import numpy as np

from numpy.random import randn, choice
from linal.utils import quadratic as quad
from linal.mp_pinv import get_mp_pinv as get_pinv
from optimization.utils import is_conv

def is_k_valid(X_ds, Y_ds, k):

    p = min([X_ds.cols(), Y_ds.cols()])

    return k < p

def is_converged(
    unn_Phi_t, 
    unn_Phi_t1, 
    unn_Psi_t, 
    unn_Psi_t1, 
    eps1, 
    eps2, 
    verbose):

    Phi_converged = is_conv(unn_Phi_t, unn_Phi_t1, eps1, verbose)
    Psi_converged = is_conv(unn_Psi_t, unn_Psi_t1, eps2, verbose)

    return Phi_converged and Psi_converged

def get_objective(X, Phi, Y, Psi):

    X_trans = np.dot(X, Phi)
    Y_trans = np.dot(Y, Psi)

    return np.linalg.norm(X_trans - Y_trans)

def get_updated_bases(X1, X2, unnormed1, normed2, S1, eta1):

    # Calculate the gradient with respect to unnormed1
    gradient = np.dot(X1.T, (np.dot(X1, unnormed1) - np.dot(X2, normed2)) ) / X1.shape[0]

    # Take a gradient step on unnormed1
    unnormed1_next = unnormed1 - eta1 * gradient

    # Normalize unnormed 1 with inversion of matrix quadratic
    normed1 = get_gram_normed(unnormed1_next, S1)

    return (unnormed1_next, normed1)

def get_init_bases(Sx, Sy, k):

    # Initialize Gaussian matrices for unnormalized bases
    unn_Phi = randn(Sx.shape[0], k)
    unn_Psi = randn(Sy.shape[0], k)

    # Normalize for initial normalized bases
    Phi = get_gram_normed(unn_Phi, Sx)
    Psi = get_gram_normed(unn_Psi, Sy) 

    return (Phi, unn_Phi, Psi, unn_Psi)

def get_gram_normed(unnormed, S):

    basis_quad = quad(unnormed, S)
    normalizer = get_pinv(
        basis_quad, energy=0.95, sqrt=True)

    return np.dot(unnormed, normalizer)
