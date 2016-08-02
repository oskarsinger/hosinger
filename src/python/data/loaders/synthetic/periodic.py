from data.loaders import AbstractDataLoader

import numpy as np

class CosineLoader(AbstractDataLoader):

    def __init__(self,
        p,
        max_rounds=1000,
        phase=0,
        amplitude=1.0,
        period=2*np.pi,
        index=0):

        self.p = p
        self.max_rounds = max_rounds
        self.phase = phase
        self.amplitude = amplitude
        self.period = period
        self.index = index

        self.transform = lambda x: x / period + phase
        self.num_rounds = 0

    def get_data(self):

        inside = self.transform(self.num_rounds)
        unscaled = np.cos(inside)
        scaled = self.amplitude * unscaled
        noise = np.random.randn(1, self.p)

        noise[0,self.index] = scaled
        self.num_rounds += 1

        return noise

    def finished(self):

        return self.num_rounds >= self.max_rounds

    def refresh(self):

        self.num_rounds = 0

    def name(self):

        return 'CosineLoader'

    def cols(self):

        return self.p

    def rows(self):

        return self.num_rounds

    def get_status(self):

        return {
            'p': self.p,
            'max_rounds': self.max_rounds,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'period': self.period,
            'index': self.index,
            'transform': self.transform,
            'num_rounds': self.num_rounds}
