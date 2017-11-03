import numpy as np

from theline.utils import get_quadratic
from models.kernels.utils import get_kernel_matrix

class SupportVectorMachineDualModel:

    def __init__(self, kernel):

        self.kernel = kernel

        if kernel is None:
            kernel = lambda x, y: np.dot(x.T, y)

        self.kernel = kernel

        self.K = None

    def get_objective(self, data, params):

        # Initialize stuff
        (X, y) = data

        if self.K is None:
            self._set_K(X)

        N = X.shape[0]
        params_y = params * y

        # Compute objective terms
        param_sum = np.sum(params)
        K_quad = get_quadratic(params_y, self.K)

        return - param_sum + 0.5 * K_quad

    def get_gradient(self, data, params, batch=None):
        
        (X, y) = data

        if self.K is None:
            self._set_K(X)

        N = X.shape[0]
        params_y = params * y
        K = None
        scale = None

        if batch is not None:
            params = params[batch,:]
            y = y[batch,:]
            K = self.K[batch,:]
            scale = self.scale[batch,:]

            if np.isscalar(batch):
                K = K[np.newaxis,:]
                params = params[:,np.newaxis]
        else:
            K = self.K
            scale = self.scale

        # Compute gradient terms
        ones = - np.ones_like(params)
        K_term = y * np.dot(K, params_y)

        scaled = (ones + K_term) / scale

        return np.fmin(params, scaled)

    def _set_K(self, X):

        self.K = get_kernel_matrix(self.kernel, X)
        self.scale = np.diag(self.K)[:,np.newaxis]
