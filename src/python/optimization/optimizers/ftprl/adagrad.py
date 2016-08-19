import numpy as np
import utils as ftprlu

from optimization.utils import get_shrunk_and_thresholded as get_st
from linal.utils import get_safe_power
from linal.svd_funcs import get_multiplied_svd

class DiagonalAdaGradOptimizer:

    def __init__(self, 
        delta=0.1,
        forget_factor=None,
        lower=None, 
        dual_avg=True, 
        verbose=False):

        self.lower = lower
        self.dual_avg = dual_avg
        self.delta = delta
        self.scale = None
        self.verbose = verbose
        self.alpha = 1
        self.beta = 1

        if forget_factor is not None:
            self.alpha = forget_factor
            self.beta = 1 - self.alpha

        self.grad = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        # Update step sizes
        if self.scale is None:
            self.scale = np.absolute(gradient)
        else:
            previous = self.alpha * get_safe_power(self.scale, 2)
            new = self.beta * get_safe_power(gradient, 2)
            self.scale = get_safe_power(previous + new, 0.5)

        # Update gradient
        self.grad = ftprlu.set_gradient(
            self.grad, gradient, self.dual_avg, self.num_rounds)
        
        return ftprlu.get_update(
            parameters, 
            eta, 
            gradient, 
            self._get_dual, 
            self._get_primal)

    def _get_dual(self, parameters):

        # Get the dual transformation
        H = self.scale + self.delta

        return H * parameters

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
            'forget_factor': self.forget_factor,
            'alpha': self.alpha,
            'beta': self.beta,
            'dual_avg': self.dual_avg,
            'grad': self.grad,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}

class FullAdaGradOptimizer:

    def __init__(self,
        delta=0.1,
        forget_factor=None,
        lower=None,
        dual_avg=True,
        sketch=False,
        verbose=False):

        self.lower = lower
        self.dual_avg = dual_avg
        self.delta = delta
        self.sketch = sketch
        self.G = None
        self.scale = None
        self.verbose = verbose
        self.alpha = 1
        self.beta = 1

        if forget_factor is not None:
            self.alpha = forget_factor
            self.beta = 1 - self.alpha

        self.grad = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        # Update step sizes
        if self.scale is None:
            self.G = np.dot(gradient, gradient.T)
        else:
            new_G = np.dot(gradient, gradient.T)
            self.G = self.alpha * self.G + self.beta * new_G

        self.scale = get_svd_power(self.G, 0.5)

        # Update gradient
        self.grad = ftprlu.set_gradient(
            self.grad, gradient, self.dual_avg, self.num_rounds)
        
        return ftprlu.get_update(
            parameters, 
            eta, 
            gradient, 
            self._get_dual, 
            self._get_primal)

    def _get_dual(self, parameters):

        # Get the dual transformation
        pd_help = self.delta * np.identity(self.scale.shape[0])
        H = self.scale + pd_help

        return np.dot(H, parameters)

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
        H_inv = get_svd_power(self.scale + self.delta, -1)

        return np.dot(H_inv, dual_update)

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'scale': self.scale,
            'sketch': self.sketch,
            'forget_factor': self.forget_factor,
            'alpha': self.alpha,
            'beta': self.beta,
            'dual_avg': self.dual_avg,
            'grad': self.grad,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
