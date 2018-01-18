import numpy as np

from theline.utils import get_quadratic
from theline.utils import get_thresholded
from models.kernels.utils import get_kernel_matrix

class Li2016SVMPlusWithMissingPrivilegedInformation:

    def __init__(self,
        c,
        gamma,
        o_kernel=None,
        p_kernel=None):

        self.c = c
        self.gamma = gamma

        if o_kernel is None:
            o_kernel = lambda x, y: np.dot(x.T, y)

        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x, y: np.dot(x.T, y)

        self.p_kernel = p_kernel

        self.K_o_missing = None
        self.K_o_not_missing = None
        self.K_p = None

        func_eval = self.o_kernel(params, data)

        return np.sign(func_eval)

    def get_objective(self, data, params):

        # Initialize stuff
        (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing) = data

        if self.K_o is None:
            self._set_Ks(X_o_missing, X_o_not_missing, X_p)

        N_missing = X_o_missing.shape[0]
        N_not_missing = X_o_not_missing.shape[0]
        N = n_missing + N_not_missing
        (alpha_missing, alpha_not_missing, beta) = (
            params[:N_missing,:], 
            params[N_missing:N,:],
            params[N:,:])
        alpha_y_missing = alpha_missing * y_missing
        alpha_y_not_missing = alpha_not_missing * y_not_missing
        alpha_beta_C = alpha_not_missing + beta - self.c

        # Compute objective terms
        alpha_sum = np.sum(alpha_missing) + np.sum(alpha_not_missing)
        K_o_missing_quad = get_quadratic(
            alpha_y_missing, 
            self.K_o_missing)
        K_o_not_missing_quad = get_quadratic(
            alpha_y_not_missing, self.K_o_not_missing)
        K_p_quad = get_quadratic(alpha_beta_C, self.K_p)

        return - alpha_sum + \
            K_o_missing_quad / 2 + \
            K_o_not_missing_quad / 2 + \
            K_p_quad / (2 * self.gamma)

    # WARNING: expects batch in ascending sorted order
    def get_gradient(self, data, params, batch=None):

        # Initialize stuff
        (X_o_missing, X_o_not_missing, X_p, y_missing, y_not_missing) = data

        if self.K_o is None:
            self._set_Ks(X_o_missing, X_o_not_missing, X_p)

        N_missing = X_o_missing.shape[0]
        N_not_missing = X_o_not_missing.shape[0]
        N = n_missing + N_not_missing
        (alpha_missing, alpha_not_missing, beta) = (
            params[:N_missing,:], 
            params[N_missing:N,:],
            params[N:,:])

        alpha_beta_C = alpha_not_missing + beta - self.c
        alpha_missing_grad = self._get_alpha_missing_grad(
            alpha_missing,
            y_missing,
            batch=None if batch is None else batch[batch < N])
        alpha_not_missing_grad = self._get_alpha_not_missing_grad(
            alpha_not_missing,
            alpha_beta_C,
            y_not_missing,
            batch=None if batch is None else batch[batch < N])
        beta_grad = self._get_beta_grad(
            beta,
            alpha_beta_C, 
            batch=None if batch is None else batch[batch >= N] - N)
        grads = [
            alpha_missing_grad, 
            alpha_not_missing_grad,
            beta_grad]
        non_zero = [g for g in grads if g.size > 0]

        return np.vstack(non_zero)

    def _get_alpha_missing_grad(self, 
        alpha, 
        y,
        batch=None):

        (K_o, K_p) = [None] * 2
        alpha_scale = None
        alpha_y = alpha * y

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            alpha = alpha[batch,:]
            y = y[batch,:]
            K_o = self.K_o_missing[batch,:]
            K_p = self.K_p[batch,:]

            if np.isscalar(batch):
                K_o = K_o[np.newaxis,:]
                K_p = K_p[np.newaxis,:]
        else:
            (K_o, K_p) = (self.K_o_missing, self.K_p)

        # Compute alpha gradient
        ones = - np.ones_like(alpha)
        K_o_term = np.dot(K_o, alpha_y) * y

        return ones + K_o_term

    def _get_beta_grad(self, beta, alpha_beta_C, batch=None):

        K_p = None
        beta_scale = None

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            beta = beta[batch,:]
            K_p = self.K_p[batch,:]

            if np.isscalar(batch):
                K_p = K_p[np.newaxis,:]
        else:
            K_p = self.K_p

        # Compute beta gradient
        return np.dot(K_p, alpha_beta_C) / self.gamma

    def _set_Ks(self, X_o_missing, X_o_not_missing, X_p):

        self.K_o_missing = get_kernel_matrix(self.o_kernel, X_o_missing)
        self.K_o_not_missing = get_kernel_matrix(self.o_kernel, X_o_not_missing)
        self.K_p = get_kernel_matrix(self.p_kernel, X_p)

    def get_projected(self, data, params):

        N_missing = data[0].shape[0]
        (alpha_missing, not_missing) = (
            params[:N_missing,:], 
            params[N_missing:,:],

        alpha_missing = get_thresholded(
            alpha_missing, upper=self.c, lower=0)
        not_missing = get_thresholded(
            not_missing, lower=0)

        return np.vstack([
            alpha_missing,
            not_missing])
