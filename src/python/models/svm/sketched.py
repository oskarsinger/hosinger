import numpy as np

from theline.utils import get_quadratic
from theline.utils import get_thresholded
from theline.sketching import GaussianSketcher as GS
from models.kernels.utils import get_kernel_matrix

class SketchedSupportVectorMachineDualModel:

    def __init__(self, c, kernel=None, sketcher=None):

        self.c = c
        
        if kernel is None:
            kernel = lambda x, y: np.dot(x.T, y)

        self.kernel = kernel
        self.sketcher = sketcher

        self.K = None

    # WARNING: assumes kernel handles arbitrary number of svs and data
    def get_prediction(self, data, params):

        (alphas, svs) = params
        alphas = self.sketcher.get_unsketched(alphas)
        eigenfunc_eval = self.kernel(svs, data)
        func_eval = np.dot(
            alphas.T, 
            np.array(kernel_evals))

        return np.sign(func_eval)

    def get_objective(self, data, params):

        # Initialize stuff
        if self.K is None:
            self._set_K(data)

        # Compute objective terms
        param_sum = np.sum(self.sketcher.get_unsketched(params))
        K_quad = get_quadratic(params, self.K)

        return - param_sum + 0.5 * K_quad

    def get_gradient(self, data, params, batch=None):
        
        if self.K is None:
            self._set_K(data)

        K = None

        if batch is not None:
            params = params[batch,:]
            K = self.K[batch,:]

            if np.isscalar(batch):
                K = K[np.newaxis,:]
                params = params[:,np.newaxis]
        else:
            K = self.K

        # TODO: fix this; it is not correct
        # Compute gradient terms
        ones = - np.ones((N, 1))
        K_term = np.dot(K, params)

        return (ones + K_term)

    def _set_K(self, data):

        (X, y) = data

        if self.sketcher is None:
            self.sketcher = GS(X.shape[0])

        K = get_kernel_matrix(self.kernel, X)
        K *= np.dot(y, y.T)
        right_sketch = self.sketcher.get_sketched(K)

        self.K = self.sketcher.get_sketched(right_sketch.T).T 

    def get_projected(self, data, params):

        unsketched = self.sketcher.get_unsketched(params)
        projected = get_thresholded(
            unsketched, upper=self.c, lower=0)

        return self.sketcher.get_sketched(projected.T).T
