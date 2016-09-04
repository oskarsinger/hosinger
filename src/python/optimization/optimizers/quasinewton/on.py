import numpy as np

from .. import utils as ou
from linal.utils import get_safe_power
from linal.svd_funcs import get_svd_power

from linal.formulas import get_woodbury_inversion

class OnlineNewtonOptimizer:

    def __init__(self,
        delta=0.1,
        forget_factor=None,
        lower=None,
        dual_avg=False,
        verbose=False):

        self.delta = delta
        self.lower = lower
        self.dual_avg = dual_avg
        self.verbose = verbose

        self.G = None
        self.alpha = 1
        self.beta = 1

        if forget_factor is not None:
            self.alpha = forget_factor
            self.beta = 1 - self.alpha

        self.grad = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        if self.G is None:
            self.G = np.dot(gradient, gradient.T)
        else:
            new_G = np.dot(gradient, gradient.T)
            self.G = self.alpha * self.G + self.beta * new_G

        self.grad = ou.get_avg_search_direction(
            self.grad, 
            gradient, 
            self.dual_avg, 
            self.num_rounds)

        return ou.get_mirror_update(
            parameters,
            eta,
            gradient,
            self._get_dual,
            self._get_primal)

    def _get_dual(self, parameters):

        pd_help = self.delta * np.identity(self.G.shape[0])
        G = self.G + pd_help

        return np.dot(G, parameters)

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

        pd_help = self.delta * np.identity(self.G.shape[0])
        G_inv = get_svd_power(self.G + pd_help, -1)

        return np.dot(G_inv, dual_update)

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'G': self.G,
            'forget_factor': self.forget_factor,
            'alpha': self.alpha,
            'beta': self.beta,
            'dual_avg': self.dual_avg,
            'grad': self.grad,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}

class SketchedOnlineNewtonOptimizer:

    def __init__(self,
        C, alpha, m,
        verbose=False):

        self.C = C
        self.alpha = alpha
        self.m = m
        self.verbose = verbose

        self.u = None
        (self.S, self.H) = [None] * 2

    def get_update(self, parameters, hessian, gradient, eta):

        if self.S is None:
            'poop'

def _fd_sketch_init(alpha, m):

    (S, H) = [None] * 2

    return (S, H)

def _fd_sketch_update(g):

    (S, H) = [None] * 2

    return (S, H)

def _oja_sketch_init(alpha, m):

    (S, H) = [None] * 2

    return (S, H)

def _oja_sketch_update(g):

    (S, H) = [None] * 2

    return (S, H)
