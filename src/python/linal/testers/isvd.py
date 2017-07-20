import numpy as np

from linal.svd import ColumnIncrementalSVD as CISVD
from whitehorses.loaders.simple import GaussianLoader as GL

class ColumnIncrementalSVDTester:

    def __init__(self, k, n, m):

        self.k = k
        self.n = n
        self.m = m

        self.loader = GL(n, m)
        # TODO: compute exact, batch SVD
        data = self.loader.get_data()
        (self.U, self.s, self.V) = np.linalg.svd(data)
        # TODO: initialize CISVD
        self.cisvd = CISVD(self.k)

    def run(self):

        for i in range(self.m):
            print('Stuff')
