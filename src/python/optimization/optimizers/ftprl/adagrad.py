import numpy as np
import utils as ftprlu

from optimization.utils import get_shrunk_and_thresholded as get_st
from linal.utils import get_safe_power
from linal.svd_funcs import get_multiplied_svd

class AdaGradOptimizer:

    def __init__(self, 
        lower=None, 
        dual_avg=True, 
        verbose=False, 
        delta=0.1):

        self.lower = lower
        self.dual_avg = dual_avg
        self.delta = delta
        self.scale = None
        self.verbose = verbose

        self.grad = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1
        self.scale = get_safe_power(
            get_safe_power(self.scale, 2) + \
                get_safe_power(gradient, 2),
            0.5)
        self.grad = ftprlu.set_gradient(
            self.grad, gradient, self.dual_avg, self.num_rounds)
        
        dual_parameters = self._get_dual(parameters)
        dual_update = dual_parameters - eta * gradient

        return self._get_primal(dual_update)

    def _get_dual(self, parameters):

        # Get the dual transformation
        H = self.scale + self.delta

        return H * primal

    def _get_primal(self, dual_update):

        if self.lower is not None:
            dus = dual_update.shape

            if len(dus) == 2 and not 1 in set(dus):
                (U, s, V) = np.linalg.svd(dual_update)
                sparse_s = get_st(s, lower=self.lower)
                dual_update = get_multiplied_svd(U, s, V)
            else:
                dual_update = get_st(
                    dual_update, lower=self.lower) 

        # Get the primal transformation
        H_inv = get_safe_power(self.scale + self.delta, -1)

        return H_inv * dual_update

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'scale': self.scale,
            'grad': self.grad,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
