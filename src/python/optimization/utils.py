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

def get_t_regged_gram(A, reg_const):

    gram = np.dot(A.T, A)
    reg_matrix = reg * np.identity(A.shape[1])

    return (gram + reg_matrix) / A.shape[0]

def get_bregman_func(get_obj, get_grad, get_ip=np.dot):

    def get_bregman(x_t, x):

        grad = get_grad(x_t)
        diff = x - x_t
        ip = get_ip(grad, diff)

        return get_obj(x) - get_obj(x_t) - ip

    return get_bregman

def get_bregman_grad_func(get_obj, get_grad, get_ip=np.dot):

    def get_bregman_grad(x_t, x):

        x_t_grad = get_grad(x_t)
        x_t_ip = get_ip(x_t_grad, x_t)
        x_grad = get_grad(x)

        return x_grad - x_t_grad - get_obj(x_t) + x_t_ip

    return get_bregman_grad
