import numpy as np

from theline.utils import get_quadratic
from theline.utils import get_thresholded
from models.kernels.utils import get_kernel_matrix

class SupportVectorMachineDualModel:

    def __init__(self, c, kernel=None):

        self.c = c

        if kernel is None:
            kernel = lambda x, y: np.dot(x.T, y)

        self.kernel = kernel

        self.K = None

    # WARNING: assumes kernel handles arbitrary number of svs and data
    def get_prediction(self, data, params):

        func_eval = self.kernel(params, data)

        return np.sign(func_eval)

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

        if batch is not None:
            params = params[batch,:]
            y = y[batch,:]
            K = self.K[batch,:]

            if np.isscalar(batch):
                K = K[np.newaxis,:]
                params = params[:,np.newaxis]
        else:
            K = self.K

        # Compute gradient terms
        ones = - np.ones_like(params)
        K_term = np.dot(K, params_y) * y

        return ones + K_term

    def _set_K(self, X):

        self.K = get_kernel_matrix(self.kernel, X)

    def get_projected(self, data, params):

        return get_thresholded(params, upper=self.c, lower=0)
