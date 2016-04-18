import numpy as np

from linal.utils import get_thresholded

def get_minibatch(A, batch_size):

    indexes = choice(
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

def get_t_regged_gram(A, reg_const):

    gram = np.dot(A.T, A)
    reg_matrix = reg * np.identity(A.shape[1])

    return (gram + reg_matrix) / A.shape[0]

def get_lp_norm_gradient(x, p):

    norm = np.linalg.norm(x, p)
    constant = norm * norm**(-1)
    vec = np.power(np.absolute(x), -1) * np.sign(x)

    return constant * vec

def get_shrunk_and_thresholded(x, lower=0):

    sign = np.sign(x)
    threshed = get_thresholded(np.absolute(x), lower=lower)

    return sign * threshed
