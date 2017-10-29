import fht

import numpy as np

from drrobert.random import rademacher
from theline.utils import RowPermutationMatrix as RPM
from scipy.stats import gengamma

# TODO: learn about Legendre polynomials and implement this one
class FastFoodInnerProductKernel:

    def __init__(self):
        pass

class FastFoodGaussianRBFKernel:

    # TODO: deal with case when n neq d
    def __init__(self, sigma, n, d, S=None, G=None, Pi=None, B=None):

        self.sigma = sigma
        self.n = n
        self.d = d

        if G is None:
            G = np.random.randn(self.d)

        self.G = G

        if S is None:
            samples = gengamma.rvs(self.d * 0.5, 2, size=self.d)
            S = samples / (np.sqrt(2) * np.thelineg.norm(self.G))

        self.S = S

        if Pi is None:
            Pi = RPM(d=self.d)

        self.Pi = Pi

        if B is None:
            B = np.rademacher(size=self.d)

        self.B = B

    def get_prediction(self, x, eigen_funcs):

        kernels = [ef(x) for ef in eigen_funcs] 

        return np.sign(sum(kernels))

    def get_eigen_function(self, x1):

        f1 = self.get_features(x1)

        return lambda x2: np.dot(
            f1.T, 
            self.get_features(x2))

    def get_kernel(self, x1, x2):

        f1 = self.get_features(x1)
        f2 = self.get_features(x2)

        return np.dot(f1.T, f2)

    def get_features(self, x):

        HBx = fht.fht(self.B * x)
        GPiHBx = self.G * self.Pi.get_transform(HBx)
        SHGPiHBx = self.S * fht.fht(GPiHBx)

        return (self.n)**(-0.5) * np.exp(1j * SHGPiHBx)
