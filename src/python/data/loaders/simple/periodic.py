import numpy as np

from gaussian import GaussianLoader

class FakePeriodicGaussianLoader:

    def __init__(self, 
        n, p, hertz,
        lazy=True,
        batch_size=None, 
        k=None, 
        mean=None):

        self.hertz = hertz
        self.loader = GaussianLoader(
            n,
            p, 
            lazy=lazy, 
            batch_size=batch_size, 
            k=k, 
            mean=None)

    def get_data(self):

        return self.loader.get_data()

    def name(self):

        return 'FakePeriodicGaussianLoader'

    def get_status(self):

        info = self.loader.get_status()

        info['hertz'] = self.hertz

        return info

    def finished(self):

        return self.loader.finished()

    def refresh(self):

        self.loader.refresh()

    def cols(self):
        
        return self.loader.cols()

    def rows(self):
        
        return self.loader.rows()

class CosineLoader:

    def __init__(self,
        p,
        max_rounds=1000,
        phase=0,
        amplitude=1.0,
        period=2*np.pi,
        index=0,
        period_noise=False,
        phase_noise=False,
        amplitude_noise=False):

        self.p = p
        self.max_rounds = max_rounds
        self.phase = phase
        self.amplitude = amplitude
        self.period = period
        self.index = index
        self.period_noise = period_noise
        self.phase_noise = phase_noise
        self.amplitude_noise = amplitude_noise

        self.num_rounds = 0

    def get_data(self):

        period = self.period
        phase = self.phase
        amplitude = self.amplitude

        if self.period_noise:
            period += np.random.randn(1)[0]

        if self.phase_noise:
            phase += np.random.randn(1)[0]

        if self.amplitude_noise:
            amplitude += np.random.randn(1)[0]

        inside = self.num_rounds / period - phase
        unscaled = np.cos(inside)
        scaled = amplitude * unscaled
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
            'num_rounds': self.num_rounds}
