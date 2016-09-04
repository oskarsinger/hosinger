import numpy as np

from optimization.utils import get_shrunk_and_thresholded as get_st
from drrobert.data_structures import FixedLengthQueue as FLQ

class PeriodicParameterProximalGradientOptimizer:

    def __init__(self,
        period, c,
        lower=None,
        verbose=False):

        self.period = period
        self.weight = period - int(period)
        self.c = c
        self.lower = lower
        self.verbose = verbose

        q_length = int(self.period)

        if self.weight > 0:
            q_length += 1

        self.window = FLQ(q_length)
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        unscaled = parameters - eta * gradient

        if self.lower is not None:
            unscaled = get_st(unscaled, lower=self.lower)

        if self.window.is_full():
            last_period = None

            if self.weight > 0:
                last = self.weight * self.window.dequeue()
                second_last = self.window.get_items()[-1]
                last_period = last + second_last
            else:
                last_period = self.window.dequeue()
            
            unscaled += self.c * last_period

        self.window.enqueue(np.copy(parameters))
        self.num_rounds += 1
        other_c = eta**(-1)

        return (other_c + other_c * self.c)**(-1) * unscaled

    def get_status(self):

        return {
            'period': self.period,
            'c': self.c,
            'verbose': self.verbose,
            'window': self.window,
            'num_rounds': self.num_rounds}

class PeriodicSignalProximalGradientOptimizer:

    def __init__(self,
        period, c,
        lower=None,
        verbose=False):

        print 'Stuff'
