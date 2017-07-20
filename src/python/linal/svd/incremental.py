import numpy as np

from linal.utils import get_multi_dot

# TODO: cite Markos paper
class RowIncrementalSVD:

    def __init__(self, k):
        pass

    def get_update(self, A):
        pass

# TODO: cite Baker 2008 paper
# TODO: consider random initialization of Q, B, W
class ColumnIncrementalSVD:

    def __init__(self, k):

        self.k = k

        self.Q = None
        self.B = None
        self.W = None
        self.m = None

    def get_update(self, A):

        if self.m is None:
            self.m = A.shape[0]
            self.Q = np.hstack([
                np.eye(self.k), 
                np.zeros((self.m - self.k, self.k))])
            self.B = np.ones(self.k)
            self.W = np.eye(self.k)

        l = A.shape[1]
        (Q_hat, B_hat) = self._get_QB_hat(A)
        W_hat = self._get_W_hat(l)
        (G_u, s, G_v) = np.linalg.svd(
            B_hat, full_matrices=False)
        
        self.B = s[:self.k]
        self.Q = np.dot(Q_hat, G_u)[:,:self.k]
        self.W = np.dot(W_hat, G_v)[:self.k,:]

        return (self.Q, self.B, self.W.T)

    def _get_W_hat(self, l, G_v):

        s = self.k + l
        W_hat = np.zero((s, s))
        W_hat[:self.k,:self.k] += self.W
        W_hat[self.k:,self.k:] += np.eye(l)

        return W_hat

    def _get_QB_hat(self, A):

        l = A.shape[1]
        s = self.k + l
        C = np.dot(self.Q.T, A)
        (Q_perp, B_perp) = np.linalg.qr(
            A - np.dot(Q, C))
        Q_hat = np.hstack([self.Q, Q_perp])
        B_hat = np.zeros((s, s))
        B_hat[:self.k,:self.k] += np.diag(self.B)
        B_hat[:self.k,self.k:] += C
        B_hat[self.k:,self.k:] += B_perp

        return (Q_hat, B_hat)
