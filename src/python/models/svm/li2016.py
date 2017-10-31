import numpy as np

from theline.utils import get_quadratic

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
        self.scale = None

    def get_objective(self, data, params):

        # Initialize stuff
        (X_o, X_p, y) = data
        N = X_o.shape[0]
        (alpha, beta) = (params[:N,:], params[N:,:])
        alpha_y = alpha * y
        alpha_beta_C = alpha + beta - self.C
        (K_o, K_p) = self._get_Ks(X_o, X_p)

        # Compute objective terms
        alpha_sum = np.sum(alpha)
        K_o_quad = get_quadratic(alpha_y, K_o)
        K_p_quad = get_quadratic(alpha_beta_C, K_p)

        return - alpha_sum + \
            0.5 * K_o_quad + \
            K_p_quad / (2 * self.gamma)
        
    def get_gradient(self, data, params, batch=None):

        # Initialize stuff
        N = int(params.shape[0] / 2)
        (alpha, beta) = (params[:N,:], params[N:,:])
        (X_o, X_p, y) = data
        (K_o, K_p) = self._get_Ks(X_o, X_p)
        alpha_y = alpha * y
        alpha_beta_C = alpha + beta - self.C

        alpha_grad = self._get_alpha_grad(
            alpha,
            alpha_beta_C,
            alpha_y,
            y,
            K_o,
            K_p,
            batch=None if batch is None else batch[batch < N])[:,np.newaxis]
        beta_grad = self._get_beta_grad(
            beta,
            alpha_beta_C, 
            K_p,
            batch=None if batch is None else batch[batch >= N])[:,np.newaxis]

        return np.vstack([alpha_grad, beta_grad])

    def _get_alpha_grad(self, 
        alpha, 
        alpha_beta_C, 
        alpha_y, 
        y,
        K_o,
        K_p,
        batch=None):

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            alpha = alpha[batch,:]
            y = y[batch,:]
            K_o = K_o[batch,:]
            K_p = K_p[batch,:]

            if np.isscalar(batch):
                K_o = K_o[np.newaxis,:]
                K_p = K_p[np.newaxis,:]

        # Compute alpha gradient
        ones = - np.ones_like(alpha) 
        K_o_term = y * np.dot(K_o, alpha_y)
        K_p_term = np.dot(K_p, alpha_beta_C)
        alpha_grad = ones + K_o_term + K_p_term / self.gamma

        # Compute scaled gradient
        beta_scale = np.diag(K_p[:,batch]) / self.gamma
        alpha_scale = np.diag(K_o[:,batch]) + beta_scale
        scaled = alpha_grad / alpha_scale

        return np.min(
            np.hstack([alpha, scaled]),
            axis=1)

    def _get_beta_grad(self, beta, alpha_beta_C, K_p, batch=None):

        # Reduce to batch if doing stochastic coordinate descent
        if batch is not None:
            beta = beta[batch,:]
            K_p = K_p[batch,:]

            if np.isscalar(batch):
                K_p = K_p[np.newaxis,:]

        # Compute beta gradient
        beta_grad = np.dot(K_p, alpha_beta_C)

        # Compute scaled gradient
        beta_scale = np.diag(K_p[:,batch]) / self.gamma
        scaled = beta_grad / beta_scale

        return np.min(
            np.hstack([beta, scaled]),
            axis=1)

    def _get_Ks(self, X_o, X_p):

        if self.K_o is None:
            self.K_o = self._get_new_K(self.o_kernel, X_o)

        if self.K_p is None:
            self.K_p = self._get_new_K(self.p_kernel, X_p)

        K_o = self.K_o
        K_p = self.K_p
        
        if self.scale is None:
            beta_scale = np.diag(K_p) / self.gamma
            alpha_scale = np.diag(K_o) + beta_scale

            self.scale = np.vstack([
                alpha_scale, 
                beta_scale])

        return (K_o, K_p)

    def _get_new_K(self, kernel, X):

        N = X.shape[0] 
        K = np.zeros((N, N))

        for n in range(N):

            X_n = X[n,:]

            for m in range(n, N):

                K_nm = kernel(X_n, X[m,:])
                K[n,m] = K_nm
                K[m,n] = K_nm

        return K
