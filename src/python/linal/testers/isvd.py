import numpy as np

from linal.svd import ColumnIncrementalSVD as CISVD
from linal.svd import get_multiplied_svd as get_ms
from whitehorses.loaders.simple import GaussianLoader as GL
from whitehorses.servers.minibatch import Batch2Minibatch as B2M

class ColumnIncrementalSVDTester:

    def __init__(self, k, n, m):

        self.k = k
        self.n = n
        self.m = m

        self.loader = GL(m, n)
        self.server = B2M(
            1, 
            data_loader=self.loader,
            random=False)

        self.data = self.loader.get_data().T

        (U, s, VT) = np.linalg.svd(self.data)
        print('s', s)

        self.U = U[:,:self.k]
        self.s = s[:self.k]
        self.VT = VT[:self.k,:]
        self.approx_data = get_ms(self.U, self.s, self.VT)
        self.cisvd = CISVD(self.k)

    def run(self):
        
        interval = int(self.m / 10)

        for t in range(self.m):
            datat = self.server.get_data().T
            (Ut, st, VTt) = self.cisvd.get_update(datat)

            if t % interval == 0 or t == self.m - 1:
                print('st', st)
                approx_data_t = get_ms(Ut, st, VTt)
                dimst = approx_data_t.shape[1]
                truncd = self.approx_data[:,:dimst]
                diff = approx_data_t - truncd
                rec_loss = np.linalg.norm(diff)**2

                print('t', t)
                print('Reconstruction loss', rec_loss)
                print('s loss', np.linalg.norm(st - self.s[:st.shape[0]])**2)
