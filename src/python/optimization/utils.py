import numpy as np

from linal.utils import get_thresholded, get_safe_power

def get_minibatch(A, batch_size):

    indexes = np.random.choice(
        np.arange(A.shape[0]), 
        replace=False, 
        size=batch_size)

    return A[indexes,:]

def is_converged(previous, current, eps, verbose):

    dist = np.linalg.norm(previous - current)

    if verbose:
        print "\tChecking for convergence"
        print "\tDistance between iterates: ", dist

    return dist < eps

def get_gram(A, reg=None):

    gram = np.dot(A.T, A)

    reg_matrix = None

    if reg is not None:
        reg_matrix = reg \
            if np.isscalar(gram) else \
            reg * np.identity(gram.shape[0])
        gram = gram + reg_matrix

    return gram

def get_lp_norm_gradient(x, p):

    vec = np.sign(x) * get_safe_power(np.absolute(x), p-1)
    constant = np.linalg.norm(x, p)**(-p+2)

    return constant * vec

def get_shrunk_and_thresholded(x, lower=0):

    sign = np.sign(x)
    threshed = get_thresholded(np.absolute(x), lower=lower)

    return sign * threshed
