import numpy as np

from theline.utils import get_quadratic
from models.kernels.utils import get_kernel_matrix

# TODO: try to account for N_o neq N_p
# TODO: figure out how to enforce bias absorbed into weights
# TODO: try to reuse some of this code since there's only a slight difference
class Li2016SVMPlus:

    def __init__(self, 
        C, 
        gamma, 
        o_kernel=None,
        p_kernel=None):

        self.C = C
        self.gamma = gamma

        if o_kernel is None:
            o_kernel = lambda x, y: np.dot(x.T, y)

        self.o_kernel = o_kernel

        if p_kernel is None:
            p_kernel = lambda x, y: np.dot(x.T, y)

        self.p_kernel = p_kernel

        self.K_o = None
        self.K_p = None
        self.alpha_scale = None
        self.beta_scale = None

    def get_objective(self, data, params):

        # Initialize stuff
        (X_o, X_p, y) = data

        if self.K_o is None:
            self._set_Ks(X_o, X_p)

        N = X_o.shape[0]
        (alpha, beta) = (params[:N,:], params[N:,:])
        alpha_y = alpha * y
        alpha_beta_C = alpha + beta - self.C

        # Compute objective terms
        alpha_sum = np.sum(alpha)
        K_o_quad = get_quadratic(alpha_y, self.K_o)
        K_p_quad = get_quadratic(alpha_beta_C, self.K_p)

        return - alpha_sum + \
            0.5 * K_o_quad + \
            K_p_quad / (2 * self.gamma)
        
    # WARNING: expects batch in ascending sorted order
    def get_gradient(self, data, params, batch=None):

        # Initialize stuff
        N = int(params.shape[0] / 2)
        (alpha, beta) = (params[:N,:], params[N:,:])
        (X_o, X_p, y) = data

        if self.K_o is None:
            self._set_Ks(X_o, X_p)

        alpha_y = alpha * y
        alpha_beta_C = alpha + beta - self.C

        alpha_grad = self._get_alpha_grad(
            alpha,
            alpha_beta_C,
            alpha_y,
            y,
            batch=None if batch is None else batch[batch < N])[:,np.newaxis]
        beta_grad = self._get_beta_grad(
            beta,
            alpha_beta_C, 
            batch=None if batch is None else batch[batch >= N] - N)[:,np.newaxis]

        return np.vstack([alpha_grad, beta_grad])

    def _get_alpha_grad(self, 
        alpha, 
        alpha_beta_C, 
        alpha_y, 
        y,
        batch=None):

        (K_o, K_p) = [None] * 2
        alpha_scale = None

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            alpha = alpha[batch,:]
            y = y[batch,:]
            K_o = self.K_o[batch,:]
            K_p = self.K_p[batch,:]
            alpha_scale = self.alpha_scale[batch,:]

            if np.isscalar(batch):
                K_o = K_o[np.newaxis,:]
                K_p = K_p[np.newaxis,:]
        else:
            (K_o, K_p) = (self.K_o, self.K_p)
            alpha_scale = self.alpha_scale

        # Compute alpha gradient
        ones = - np.ones_like(alpha)
        K_o_term = y * np.dot(K_o, alpha_y)
        K_p_term = np.dot(K_p, alpha_beta_C)
        alpha_grad = ones + K_o_term + K_p_term / self.gamma

        # Compute scaled gradient
        scaled = alpha_grad / alpha_scale

        return np.fmin(alpha, scaled)

    def _get_beta_grad(self, beta, alpha_beta_C, batch=None):

        K_p = None
        beta_scale = None

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            beta = beta[batch,:]
            K_p = self.K_p[batch,:]
            beta_scale = self.beta_scale[batch,:]

            if np.isscalar(batch):
                K_p = K_p[np.newaxis,:]
        else:
            K_p = self.K_p
            beta_scale = self.beta_scale

        # Compute beta gradient
        beta_grad = np.dot(K_p, alpha_beta_C)

        # Compute scaled gradient
        scaled = beta_grad / beta_scale

        return np.fmin(beta, scaled)

    def _set_Ks(self, X_o, X_p):

        self.K_o = get_kernel_matrix(self.o_kernel, X_o)
        self.K_p = get_kernel_matrix(self.p_kernel, X_p)
        self.beta_scale = np.diag(self.K_p) / self.gamma
        self.alpha_scale = np.diag(self.K_o) + self.beta_scale
