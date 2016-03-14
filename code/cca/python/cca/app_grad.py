import numpy as np

from numpy.random import randn, choice
from linal.utils import quadratic as quad, get_svd_invert

def get_app_grad_decomp(X, Y, k, eta1, eta2, epsilon1, epsilon2, reg, batch_size=None):

    (n1, p1) = X.shape
    (n2, p2) = Y.shape
    p = min([p1, p2])
    stochastic = False

    if not n1 == n2:
        raise ValueError(
            'Data matrices must have same number of data points.')

    if k > p:
        raise ValueError(
            'The value of k must be less than or equal to the minimum of the' +
            ' number of columns of X and Y.')

    if batch_size is not None:
        if batch_size < k:
            raise ValueError(
                'Batch size must be greater than or equal to the value of k.')
        else:
            stochastic = True

    Sx = None
    Sy = None

    print "Getting Sx and Sy"

    # Empirical second moment matrices for X and Y
    if stochastic:
        Sx = _get_minibatch_S(X, batch_size, reg)
        Sy = _get_minibatch_S(Y, batch_size, reg)
    else:
        Sx = _get_regged_gram(X, reg) / n1
        Sy = _get_regged_gram(Y, reg) / n1

    print "Getting initial basis estimates"

    # Randomly initialize normalized and unnormalized canonical bases for 
    # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
    (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t) = _get_init_bases(Sx, Sy, k)
    (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

    converged = False

    i = 1

    while not converged:

        print "Iteration:", i

        eta1_current = eta1 / i**0.5
        eta2_current = eta2 / i**0.5
        i = i + 1

        if stochastic:
            print "\tGetting minibatch Sx and Sy"

            Sx = _get_minibatch_S(X, batch_size, reg)
            Sy = _get_minibatch_S(Y, batch_size, reg)

        print "\tGetting updated basis estimates"

        # Get basis updates for both X and Y's canonical bases, normed and unnormed
        (unn_Phi_t1, Phi_t1) = _get_updated_bases(
            X, Y, unn_Phi_t, Psi_t, Sx, k, eta1_current)
        (unn_Psi_t1, Psi_t1) = _get_updated_bases(
            Y, X, unn_Psi_t, Phi_t, Sy, k, eta2_current)

        print "\tChecking for convergence"

        # Check if error is below tolerance threshold
        converged = _is_converged(unn_Phi_t, unn_Phi_t1, epsilon1) and \
            _is_converged(unn_Psi_t, unn_Psi_t1, epsilon2)

        # Update state
        (unn_Phi_t, Phi_t, unn_Psi_t, Psi_t) = (unn_Phi_t1, Phi_t1, unn_Psi_t1, Psi_t1)

    return (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t)

def _get_minibatch_S(A, batch_size, reg):

    indexes = choice(
        np.arange(A.shape[0]), replace=False, size=batch_size)
    A_t = A[indexes,:]

    return _get_regged_gram(A_t, reg) / batch_size

def _get_regged_gram(A, reg):

    gram = np.dot(A.T, A)
    reg_matrix = reg * np.identity(A.shape[1])

    return gram + reg_matrix

def _is_converged(unnormed, unnormed_next, epsilon):

    # Calculate distance between current and previous timesteps' bases under 
    # Frobenius norm
    distance = np.linalg.norm(unnormed - unnormed_next)

    print "\t\tDistance:", distance

    return distance < epsilon

def _get_updated_bases(X1, X2, unnormed1, normed2, S1, k, eta1):

    n = X1.shape[0]

    # Calculate the gradient with respect to unnormed1
    gradient = np.dot(X1.T, (np.dot(X1, unnormed1) - np.dot(X2, normed2)) ) / n

    # Take a gradient step on unnormed1
    unnormed1_next = unnormed1 - eta1 * gradient

    # Normalize unnormed 1 with inversion of matrix quadratic
    normed1 = _get_quad_normed(unnormed1_next, S1)

    return (unnormed1_next, normed1)

def _get_init_bases(Sx, Sy, k):

    # Initialize Gaussian matrices for unnormalized bases
    unn_Phi = randn(Sx.shape[0], k)
    unn_Psi = randn(Sy.shape[0], k)

    # Normalize for initial normalized bases
    Phi = _get_quad_normed(unn_Phi, Sx)
    Psi = _get_quad_normed(unn_Psi, Sy) 

    return (Phi, unn_Phi, Psi, unn_Psi)

def _get_quad_normed(unnormed, S):

    normalizer = get_svd_invert(
        quad(unnormed, S), random=False, power=-0.5)

    return np.dot(unnormed, normalizer)
