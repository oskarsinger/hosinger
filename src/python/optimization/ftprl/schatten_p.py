from optimization.comid import AbstractFTPRL
from optimization.utils import get_lp_norm_gradient as get_lpn_grad

import numpy as np

class SchattenPCOMID(AbstractFTPRL):

    def __init__(self, lower=None, dual_avg=None, p=2):

        super(SchattenPCOMID, self).__init__(lower, dual_avg)
       
        self.p = p
        self.q = float(p)/(p - 1)

    def _get_dual(self, primal):

        return get_lpn_grad(primal, self.p)

    def _get_primal(self, dual):

        return get_lpn_grad(dual, self.q)

    def get_status(self):

        init = super(SchattenPCOMID, self).get_status()

        return init + {
            'p': self.p,
            'q': self.q}

