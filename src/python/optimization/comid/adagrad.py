from optimization.comid import AbstractCOMID
from linal.utils import multi_dot, get_safe_power

import numpy as np

class MatrixAdaGradCOMID(AbstractCOMID):

    def __init__(self, delta=0.1):

        self.U = None
        self.s = None
        self.V = None
        self.scale = None

        self.delta = delta

    def _update_state(parameters, gradient, eta):

        # Get gradient's singular values
        grad_s = np.linalg.svd(gradient, compute_uv=False)

        # Update diagonals of transformation matrix
        self.scale = get_safe_power(
            get_safe_power(self.scale, 2) + get_safe_power(grad_s, 2),
            0.5)

    def _get_dual(self):

        # Get the dual transformation's matrix
        H = self.scale + self.delta

        return H * self.s

    def _get_primal(self):

        return get_safe_power(H, -1) * self.s

    def get_status(self):

        return {
            'U': self.U,
            's': self.s,
            'V': self.V,
            'sparse': self.sparse,
            'delta': self.delta,
            'scale': self.scale,
            'lambda': self.lam}


