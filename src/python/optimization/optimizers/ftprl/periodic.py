from optimization.optimizers.ftprl import AbstractMatrixFTPRLOptimizer
from drrobert.data_structures import FixedLengthQueue as FLQ

import numpy as np

class PeriodicParameterMirrorDescent:

    def __init__(self,
        period, c,
        lower=None,
        verbose=False):

        super(PeriodicMirrorDescent, self).__init__(lower, dual_avg)

        self.period = period
        self.c = c
        self.window = FLQ(self.period)

    def get_update(self, parameters, gradient, eta):

        self.window.enqueue(np.copy(parameters))

        unscaled = parameters

        if self.window.get_length() == self.period:
            unscaled += self.c * self.window.dequeue()
        elif self.window.get_length() > self.period:
            raise Exception(
                'PeriodicParameterMirrorDescent queue too long.')

        unscaled -= eta * gradient

        return (eta + eta * self.c)**(-1) * unscaled

    def get_status(self):

        status = super(MatrixAdaGrad, self).get_status()

        status['period'] = self.period
        status['c'] = self.c
        status['window'] = self.window

        return status

class PeriodicSignalMirrorDescent:

    def __init__(self,
        period, c,
        lower=None,
        verbose=False):

        print 'Stuff'
