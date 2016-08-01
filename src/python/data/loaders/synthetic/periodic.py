from data.loaders import AbstractDataLoader

import numpy as np

class CosineLoader(AbstractDataLoader):

    def __init__(self,
        max_rounds=1000,
        phase=0,
        amplitude=1.0,
        period=2*np.pi):

        self.max_rounds = max_rounds
        self.phase = phase
        self.amplitude = amplitude
        self.period = period

        self.transform = lambda x: x / period + phase
        self.num_rounds = 0

    def get_data(self):

        inside = self.transform(self.num_rounds)
        unscaled = np.cos(inside)

        self.num_rounds += 1

        return self.amplitude * unscaled

    def finished(self):

        return self.num_rounds >= self.max_rounds

    def refresh(self):

        self.num_rounds = 0

    def name(self):

        return 'CosineLoader'

    def cols(self):

        return 1

    def rows(self):

        return self.num_rounds

    def get_status(self):

        return {
            'max_rounds': self.max_rounds,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'period': self.period,
            'transform': self.transform,
            'num_rounds': self.num_rounds}
