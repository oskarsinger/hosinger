import numpy as np

from math import floor, ceil

class AlTestRampLoader:

    def __init__(self,
        freq=1.0/60,
        period=3600*24,
        num_periods=8,
        prop_jitter=0.0,
        supp1=0.1, supp2=0.2,
        sigma_s1=0.5, sigma_s2=0.5,
        sigma_n=0.001,
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
        self.sigma_n = sigma_n
        self.s1 = s1
        self.s2 = s2
        self.random_scaling = random_scaling

        self.T = int(floor(self.freq * self.period))
        self.num_points = self.T * self.num_periods

        rand_offsets = np.around(
            (np.random.uniform((N,)) - 0.5) * \
            self.prop_jitter * \
            self.T)
        signal1 = np.arange(
            int(self.T * self.supp1))[:,np.newaxis]
        signal1 /= np.linalg.norm(signal1)
        signal2 = np.ones(
            (int(self.T * self.supp2), 1))
        signal2 /= np.linalg.norm(signal2)

        self.TS1 = self._get_TS(
            signal1,
            self.sigma_s1,
            self.s1,
            self.supp1,
            rand_offsets)
        self.TS2 = self._get_TS(
            signal2,
            self.sigma_s2,
            self.s2,
            self.supp2,
            rand_offsets)

    def get_data(self):

        return (self.TS1, self.TS2)

    def _get_cycle(self, 
        signal, 
        sigma, 
        s, 
        supp, 
        rand_offsets):

        periods = []

        for k in xrange(self.N):

            ro = random_offsets[k]
            zs = np.zeros(
                (int(ceil(self.T * (1-supp))+ro),1))
            period = np.copy(signal)

            if self.random_scaling:
                scaling = sigma * np.abs(np.random.randn())
                period = scaling * period

            periods.extend([period, zs])

        full = np.vstack(periods)

        if full.shape[0] > self.num_points:
            full = full[:self.num_points,:]
        else:
            padding = np.zeros((
                self.num_points - full.shape[0],1))
            full = np.hstack([full, padding])

        noise = np.random.randn(self.num_points, 1)

        return cycle + self.sigma_n * noise
