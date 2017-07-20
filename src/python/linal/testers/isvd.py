import numpy as np

from linal.svd import ColumnIncrementalSVD as CISVD
from whitehorses.loaders.simple import GaussianLoader as GL
from whitehorses.servers.minibatch import Batch2Minibatch as B2M

class ColumnIncrementalSVDTester:

    def __init__(self, k, n, m):

        self.k = k
        self.n = n
        self.m = m

        self.loader = GL(n, m)
        self.server = B2M(
            1, data_loader=self.loader)

        data = self.loader.get_data()
        (U, s, V) = np.linalg.svd(data)

        self.U = U[:,:self.k]
        self.s = s[:self.k]
        sefl.V = V[:self.k,:]
        self.cisvd = CISVD(self.k)

    def run(self):
        
        interval = int(self.m / 10)

        for t in range(self.m):
            data = self.server.get_data()
            (Ut, st, Vt) = self.cisvd.get_update(data)
            U_loss = np.linalg.norm(Ut - self.U)**2
            s_loss = np.linalg.norm(st - self.s)**2
            V_loss = np.linalg.norm(Vt - self.V)**2

            if t % interval == 0:
                print('t', t)
                print('U_loss', U_loss)
                print('s_loss', s_loss)
                print('V_loss', V_loss)
