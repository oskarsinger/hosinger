from optimization.optimizers.ftprl import AbstractMatrixFTPRLOptimizer

import numpy as np

class PeriodicMirrorDescent(AbstractMatrixFTPRLOptimizer):

    def __init__(self,
        period,
        lower=None,
        dual_avg=False,
        verbose=False):

        super(PeriodicMirrorDescent, self).__init__(lower, dual_avg)

        self.period = period

    def _get_dual(self, primal):

        return primal

    def _get_primal(self, dual):

        return dual

    def get_status(self):

        status = super(MatrixAdaGrad, self).get_status()

        status['period'] = self.period

        return status
