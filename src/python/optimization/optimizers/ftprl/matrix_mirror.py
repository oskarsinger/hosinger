import numpy as np
from .. import utils as ou

from optimization.utils import get_lp_norm_gradient as get_lpn_grad
from optimization.utils import get_shrunk_and_thresholded as get_st
from linal.svd_funcs import get_multiplied_svd

class SchattenPCOMIDOptimizer:

    def __init__(self, 
        lower=None, 
        dual_avg=False, 
        p=2, 
        verbose=False):

        self.lower = lower
        self.dual_avg = dual_avg
        self.p = p
        self.q = float(p)/(p - 1)
        self.verbose = verbose

        self.grad = None
        self.num_rounds = 0
        (self.U, self.s, self.V) = [None] * 3

    def get_update(self, parameters, gradient, eta):

        if any([x is None for x in [self.u, self.s, self.V]]):
            (self.U, self.s, self.V) = np.linalg.svd(parameters)

        self.num_rounds += 1
        self.grad = ou.get_avg_search_direction(
            self.grad, 
            gradient, 
            self.dual_avg, 
            self.num_rounds)

        return ou.get_mirror_update(
            parameters,
            eta,
            self.grad,
            self._get_dual,
            self._get_primal)

    def _get_dual(self, parameters):

        dual_s = get_lpn_grad(self.s, self.p)

        return get_multiplied_svd(self.U, dual_s, self.V)

    def _get_primal(self, dual_update):

        (self.U, dual_s, self.V) = np.linalg.svd(dual_update)

        if self.lower is not None:
            dual_s = get_st(dual_s, lower=self.lower)

        self.s = get_lpn_grad(dual_s, self.q)

        return get_multiplied_svd(self.U, self.s, self.V)

    def get_status(self):

        return {
            'num_rounds': self.num_rounds,
            'U': self.U,
            's': self.s,
            'V': self.V,
            'grad': self.grad,
            'lower': self.lower,
            'dual_avg': self.dual_avg,
            'verbose': self.verbose,
            'p': self.p,
            'q': self.q}
