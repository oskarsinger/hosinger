from optimization.ftprl import AbstractFTPRL
from linal.utils import multi_dot, get_safe_power

import numpy as np

class MatrixAdaGrad(AbstractFTPRL):

    def __init__(self, lower=None, dual_avg=None, delta=0.1):

        super(MatrixAdaGrad, self).__init__(lower, dual_avg)

        self.delta = delta

        self.scale = None

    # Overrides super class to update adaptive prox func's parameters
    def _set_gradient(self, gradient):

        if self.grad is None:
            self.grad = np.zeros_like(gradient)

        if self.dual_avg: 
            # Get averaged gradient if desired
            self.grad = get_ra(
                self.grad, gradient, self.num_rounds)
        else:
            # Otherwise, get current gradient
            self.grad = gradient

        # Get gradient's singular values
        grad_s = np.linalg.svd(self.grad, compute_uv=False)

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


