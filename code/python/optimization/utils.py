import numpy as np

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
