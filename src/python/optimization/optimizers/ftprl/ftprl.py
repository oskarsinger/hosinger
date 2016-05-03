from abc import abstractmethod

from optimization.utils import get_shrunk_and_thresholded as get_st
from optimization.optimizers import AbstractOptimizer
from linal.svd_funcs import get_multiplied_svd
from global_utils.arithmetic import get_running_avg as get_ra

class AbstractMatrixFTPRLOptimizer(AbstractOptimizer):

    def __init__(self, lower=None, dual_avg=True):

        self.sparse = self.lower is not None

        if self.sparse:
            if lower < 0:
                raise ValueError(
                    'Parameter lower should have non-negative value.')

        self.lower = lower
        self.dual_avg = dual_avg

        self.grad = None
        self.num_rounds = 0
        (self.U, self.s, self.V) = [None] * 3

    def get_update(self, parameters, gradient, eta):

        if any([x is None for x in [self.U, self.s, self.V]]):
            (self.U, self.s, self.V) = np.linalg.svd(parameters)
            
        self.num_rounds += 1

        self._set_gradient(gradient)

        # Get dual parameters
        dual_params = self._get_dual_parameters()

        # Take gradient step in dual space
        dual_update = dual_params - eta * self.grad

        # Map back into primal space
        return self._get_primal_parameters(dual_update)

    def _get_primal_parameters(self, dual_update):

        # Update singular vectors of cached SVD
        (self.U, dual_s, self.V) = np.linalg.svd(dual_update)

        # Map singular values back into primal space
        self.s = self._get_primal(dual_s)

        return get_multiplied_svd(self.U, self.s, self.V)

    def _get_dual_parameters(self):

        # Map singular values into dual space
        dual_s = self._get_dual(self.s)

        # Shrink and threshold if sparsity desired
        if self.sparse:
            dual_s = get_st(dual_s, self.lower)

        # Re-multiply with dual singular values
        return get_multiplied_svd(self.U, dual_s, self.V)

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

    def get_status(self):

        return {
            'U': self.U,
            's': self.s,
            'V': self.V,
            'sparse': self.sparse,
            'lower': self.lower,
            'dual_avg': self.dual_avg,
            'gradient': self.grad,
            'num_rounds': self.num_rounds}

    @abstractmethod
    def _get_dual(self, primal):
        pass

    @abstractmethod
    def _get_primal(self, dual):
        pass


