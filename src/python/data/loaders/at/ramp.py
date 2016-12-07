import numpy as np

from math import floor

class AlTestRampLoader:

    def __init__(self,
        freq=1.0/60,
        period=3600*24,
        num_periods=8,
        prop_jitter=0.0,
        supp1=0.1, supp2=0.2,
        sigma_s1=0.5, sigma_s2=0.5,
        s1=6, s2=6,
        random_scaling=True):

        self.freq = freq
        self.period = period
        self.num_periods = num_periods
        self.prop_jitter = prop_jitter
        self.supp1 = supp1
        self.supp2 = supp2
        self.sigma_s1 = sigma_s1
        self.sigma_s2 = sigma_s2
        self.s1 = s1
        self.s2 = s2
        self.random_scaling = random_scaling

        self.T = int(floor(self.freq * self.period))
        self.num_points = self.T * self.num_periods

        rand_offsets = np.around(
            (np.random.uniform((N,)) - 0.5) * \
            self.prop_jitter * \
            self.T)
        signal1 = 'Poop'
        signal2 = 'Poop'

        self.cycle1 = self._get_cycle(
            signal1,
            self.sigma_s1,
            self.s1,
            self.supp1,
            rand_offsets)
        self.cycle2 = self._get_cycle()
            signal2,
            self.sigma_s2,
            self.s2,
            self.supp2,
            rand_offsets)

    def get_data(self):

        return (self.cycle1, self.cycle2)

    def _get_cycle(self, 
        signal, 
        sigma, 
        s, 
        supp, 
        rand_offsets):

        cycle = None

        for k in xrange(self.N):

            if self.random_scaling:
                cycle = 'Poop'
            else:
                cycle = 'Poop'
