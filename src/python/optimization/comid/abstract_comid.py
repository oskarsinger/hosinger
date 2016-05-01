from abc import ABCMeta, abstractmethod

from linal.svd_funcs import get_multiplied_svd
from optimization.utils import get_shrunk_and_thresholded as get_st

class AbstractMatrixCOMID:

    def get_comid_update(self, parameters, gradient, eta, lower=None):

        if any([x is None for x in [self.U, self.s, self.V, self.scale]]):
            (self.U, self.s, self.V) = np.linalg.svd(parameters)
            self.scale = np.zeros_like(self.s)

        # If necessary, update mutable state of implicit update
        self._update_state(parameters, gradient, eta)

        # Map singular values into dual space
        dual_s = self._get_dual(self.s)

        # Shrink and threshold if shrinkage constant provided
        if lower is not None:
            dual_s = get_st(dual_s, lower)

        dual_params = get_multiplied_svd(self.U, dual_s, self.V)

        # Take gradient step in dual space
        dual_update = dual_params - eta * gradient

        # Update cached SVD of dual parameters
        (self.U, self.s, self.V) = np.linalg.svd(dual_update)

        # Map singular values back into primal space
        self.s = self._get_primal(self.s)

        return get_multiplied_svd(self.U, self.s, self.V)

    @abstractmethod
    def _update_state(parameters, gradient, eta):
        pass

    @abstractmethod
    def _get_dual(self):
        pass

    @abstractmethod
    def _get_primal(self):
        pass

    @abstractmethod
    def get_status(self):
        pass
