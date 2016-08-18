import numpy as np

from linal.utils import get_safe_power
from linal.svd_funcs import get_multiplied_svd
from linal.formulas import get_woodbury_inversion

class SketchedOnlineNewtonOptimizer:

    def __init__(self,
        C, alpha, m,
        verbose=False)

        self.C = C
        self.alpha = alpha
        self.m = m
        self.verbose = verbose

    def get_update(self, parameters, hessian, gradient, eta):

        print 'Stuff'
