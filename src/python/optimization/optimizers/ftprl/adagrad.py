from optimization.optimizers.ftprl import AbstractMatrixFTPRLOptimizer
from linal.utils import get_safe_power

import numpy as np

class MatrixAdaGrad(AbstractMatrixFTPRLOptimizer):

    def __init__(self, lower=None, dual_avg=True, delta=0.1):

        super(MatrixAdaGrad, self).__init__(lower, dual_avg)

        self.delta = delta

        self.scale = None

    # Overrides super class to update adaptive prox func's parameters
    def _set_gradient(self, gradient):

        super(MatrixAdaGrad, self)._set_gradient(gradient)

        # Get gradient's singular values
        grad_s = np.linalg.svd(self.grad, compute_uv=False)

        if self.scale is None:
            self.scale = np.zeros_like(grad_s)

        # Update diagonals of transformation matrix
        self.scale = get_safe_power(
            get_safe_power(self.scale, 2) + get_safe_power(grad_s, 2),
            0.5)

    def _get_dual(self, primal):

        # Get the dual transformation
        H = self.scale + self.delta

        return H * primal

    def _get_primal(self, dual):

        # Get the primal transformation
        H_inv = get_safe_power(self.scale + self.delta, -1)

        return H_inv * dual

    def get_status(self):

        return init + {
            'delta': self.delta,
            'scale': self.scale}


