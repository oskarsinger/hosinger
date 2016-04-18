from optimization.bregman import get_lp_bregman_div_and_grad as get_lp_breg
from optimization.utils import get_shrunk_and_thresholded as get_s_and_t

from linal.utils import multi_dot, get_safe_power

class SchattenPCOMID(AbstractCOMID):

    def __init__(self, p=2, lam=0.1, s_and_t=False):

        self.U = None
        self.s = None
        self.V = None
        
        breg_div, breg_grad = get_lp_breg(p)

        self.breg_div = breg_div
        self.breg_grad = breg_grad
        self.p = p
        self.lam = lam

    def get_comid_update(self, parameters, gradient, eta):

        if self.U is None or self.V is None or self.V is None:
            (self.U, self.s, self.V) = np.linalg.svd(parameters)

        # Take mirror descent gradient step
        gradient_s = np.diag(self.breg_grad(self.s))
        breg_grad_params = multi_dot([self.U, gradient_s, self.V])
        gradient_step = breg_grad_params - eta * gradient
        
        # Updating cached SVD
        (self.U, self.s, self.V) = np.linalg.svd(gradient_step)

        # Take composite minimization step
        s = get_s_and_t(self.s, eta * self.lam) \
            if self.s_and_t else \
            self.s
        gradient_s = self.breg_grad(s)
        self.s = get_safe_power(gradient_s, -1)
        
        return [np.copy(A) 
                for A in [self.U, np.diag(self.s), self.V]]

    def get_status(self):

        return {
            'U': self.U,
            's': self.s,
            'V': self.V,
            'breg_div': self.breg_div,
            'breg_grad': self.breg_grad,
            's_and_t': self.s_and_t,
            'p': self.p,
            'lambda': self.lam}

