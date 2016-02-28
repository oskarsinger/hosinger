import numpy as np

from numpy.random import randn
from linal.utils import quadratic, get_svd_invert

def get_batch_app_grad_decomp(X, Y, k, eta1, eta2, epsilon1, epsilon2):

    (n1, p1) = X.shape
    (n2, p2) = Y.shape

    if not n1 == n2:
        raise ValueError(
            'Data matrices must have same number of data points.')

    n = n1
    p =  min([p1,p2])

    # Empirical second moment matrices for X and Y
    Sx = np.dot(X.T, X) / n
    Sy = np.dot(Y.T, Y) / n

    # Randomly initialize normalized and unnormalized canonical bases for 
    # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
    (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t) = _init_bases(p, p1, p2)
    (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

    converged = False

    while not converged:

        # Get basis updates for both X and Y's canonical bases, normed and unnormed
        (unn_Phi_t1, Phi_t1) = _get_updated_bases(X, Y, unn_Phi_t, Psi_t, Sx, k, eta1)
        (unn_Psi_t1, Psi_t1) = _get_updated_bases(Y, X, unn_Psi_t, Phi_t, Sy, k, eta2)

        # Check if error is below tolerance threshold
        converged = _is_converged(unn_Phi_t, unn_Phi_t1, epsilon1) and \
            _is_converged(unn_Psi_t, unn_Psi_t1, epsilon2)

        # Update state
        (unn_Phi_t, Phi_t, unn_Psi_t, Psi_t) = (unn_Phi_t1, Phi_t1, unn_Psi_t1, Psi_t1)

    return (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t)

def _is_converged(unnormed, unnormed_next, epsilon):

    # Calculate distance between current and previous timesteps' bases under 
    # Frobenius norm
    distance = np.linalg.norm(unnormed - unnormed_next)

    return distance < epsilon

def _get_updated_bases(X1, X2, unnormed1, normed2, S1, k, eta1):

    n = X1.shape[0]

    # Calculate the gradient with respect to unnormed1
    gradient = np.dot(X1.T, (np.dot(X1, unnormed1) - np.dot(X2, normed2)) ) / n

    # Take a gradient step on unnormed1
    unnormed1_next = unnormed1 - eta1 * gradient

    # Take randomized SVD of quadratic of unnormed1 with coefficient matrix S1
    unnormed1_quad_invert = get_svd_invert(quadratic(unnormed1_next, S1), random=False)

    # Normalize unnormed 1 with inversion of quadratic
    normed1 = np.dot(unnormed1_next, unnormed1_quad_invert)

    return (unnormed1_next, normed1)

def _init_bases(p, p1, p2):
    
    # Initialize Gaussian matrices for bases
    Phi = randn(p1, p)
    unn_Phi = randn(p1, p)
    Psi = randn(p2, p)
    unn_Psi = randn(p2, p)

    return (Phi, unn_Phi, Psi, unn_Psi)
