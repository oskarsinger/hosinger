from optimization.utils import get_shrunk_and_thresholded as get_st
from optimization.utils import get_lp_norm_gradient as get_lpn_grad

from linal.utils import multi_dot

class SchattenPCOMID(AbstractCOMID):

    def __init__(self, p=2, lam=0.1, sparse=False):

        self.U = None
        self.s = None
        self.V = None
        
        self.p = p
        self.q = float(p)/(p - 1)
        self.get_dual = get_lpn_grad(self.p)
        self.get_primal = get_lpn_grad(self.q)
        self.lam = lam

    def get_comid_update(self, parameters, gradient, eta):

        if self.U is None or self.V is None or self.V is None:
            (self.U, self.s, self.V) = np.linalg.svd(parameters)

        # Map singular values into dual space
        dual_s = np.diag(self.get_dual(self.s))
        dual_params = multi_dot([self.U, dual_s, self.V])

        # Take gradient step in dual space
        dual_update = dual_params - eta * gradient
        
        # Update cached SVD of dual parameters
        (self.U, self.s, self.V) = np.linalg.svd(dual_update)

        # Shrink and threshold if sparsity is desired
        st = get_st(self.s, eta * self.lam) if self.sparse else self.s

        # Map singular values back into primal space
        self.s = self.get_primal(st)
        
        return multi_dot([self.U, np.diag(self.s), self.V])

    def get_status(self):

        return {
            'U': self.U,
            's': self.s,
            'V': self.V,
            'sparse': self.sparse,
            'p': self.p,
            'q': self.q,
            'get_dual': self.get_dual,
            'get_primal': self.get_primal,
            'lambda': self.lam}

