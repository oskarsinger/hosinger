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
            mod_cond = t % interval == 0
            end_cond = t == self.m - 1
            and_right = mod_cond or end_cond

            if t > self.k and and_right:
                print('t', t)

                U_loss = np.linalg.norm(Ut - self.U)**2
                s_loss = np.linalg.norm(st - self.s)**2
                VT_comp = self.VT[:,:VTt.shape[1]]
                VT_loss = np.linalg.norm(VTt - VT_comp)**2

                print('U loss', U_loss)
                print('s loss', s_loss)
                print('VT_loss', VT_loss)
