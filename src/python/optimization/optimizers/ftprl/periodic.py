from optimization.optimizers.ftprl import AbstractMatrixFTPRLOptimizer
from drrobert.data_structures import FixedLengthQueue as FLQ

import numpy as np

class PeriodicParameterMirrorDescent:

    def __init__(self,
        period, c,
        verbose=False):

        self.period = period
        self.weight = period - int(period)
        self.c = c
        self.verbose = verbose

        q_length = int(self.period) + 1
        self.window = FLQ(q_length)
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        unscaled = parameters - eta * gradient

        if self.window.get_length() == int(self.period) + 1:
            last = self.weight * self.window.dequeue()
            second_last = self.window.get_items()[-1]
            last_period = last + second_last
            unscaled += self.c * last_period

        self.window.enqueue(np.copy(parameters))
        self.num_rounds += 1

        return (eta + eta * self.c)**(-1) * unscaled

    def get_status(self):

        return {
            'period': self.period,
            'c': self.c,
            'verbose': self.verbose,
            'window': self.window,
            'num_rounds': self.num_rounds}

class PeriodicSignalMirrorDescent:

    def __init__(self,
        period, c,
        lower=None,
        verbose=False):

        print 'Stuff'
