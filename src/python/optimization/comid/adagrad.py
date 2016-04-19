from optimization.comid import AbstractCOMID
from optimization.utils import get_shrunk_and_thresholded as get_st
from linal.utils import multi_dot, get_safe_power
from linal.svd_funcs import get_multiplied_svd

import numpy as np

class AdaGradCOMID(AbstractCOMID):

    def __init__(self, delta=0.1):

        self.U = None
        self.s = None
        self.V = None
        self.scale = None

        self.delta = delta

    def get_comid_update(self, parameters, gradient, eta):

        if any([x is None for x in [self.U, self.s, self.V, self.scale]]):
            (self.U, self.s, self.V) = np.linalg.svd(parameters)
            self.scale = np.zeros_like(self.s)

        # Update proximal function parameters
        grad_s = np.linalg.svd(gradient, compute_uv=False)
        self.scale = get_safe_power(
            get_safe_power(self.scale, 2) + get_safe_power(grad_s, 2),
            0.5)
        H = self.scale + self.delta

        # Map singular values into dual space
        dual_s = H * self.s
        dual_params = get_multiplied_svd(self.U, dual_s, self.V)

        # Take gradient step in dual space
        dual_update = dual_params - eta * gradient

        # Update cached SVD of dual parameters
        (self.U, self.s, self.V) = np.linalg.svd(dual_update)

        # Map singular values back into primal space
        self.s = get_safe_power(H, -1) * self.s

        return get_multiplied_svd(self.U, self.s, self.V)

    def get_status(self):

        return {
            'U': self.U,
            's': self.s,
            'V': self.V,
            'sparse': self.sparse,
            'delta': self.delta,
            'scale': self.scale,
            'lambda': self.lam}


