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
        self.l = 0
        self.num_rounds = 0

    def get_update(self, A):

        if self.l == 0:
            self.m = A.shape[0]

        pre_trunc_Q = None
        pre_trunc_B = None
        pre_trunc_W = None
        lt = A.shape[1]

        if self.num_rounds == 0:
            (Q, B) = np.linalg.qr(A)
            (U, pre_trunc_B, WT) = np.linalg.svd(B)
            pre_trunc_Q = np.dot(Q, U)
            pre_trunc_W = WT.T
        else:
            (Q_hat, B_hat) = self._get_QB_hat(A)
            W_hat = self._get_W_hat(lt)
            (G_u, pre_trunc_B, G_vT) = np.linalg.svd(
                B_hat, full_matrices=False)
            pre_trunc_Q = np.dot(Q_hat, G_u)
            pre_trunc_W = np.dot(W_hat, G_vT.T)
        
        self.B = pre_trunc_B[:self.k]
        self.Q = pre_trunc_Q[:,:self.k]
        self.W = pre_trunc_W[:,:self.k]
        self.l += lt
        self.num_rounds += 1

        return (self.Q, self.B, self.W.T)

    def _get_W_hat(self, l):

        (Wn, Wm) = self.W.shape
        W_hat = np.zeros(
            (Wn + l, Wm + l))
        W_hat[:Wn,:Wm] += self.W
        W_hat[Wn:,Wm:] += np.eye(l)

        return W_hat

    def _get_QB_hat(self, A):

        lt = A.shape[1]
        kt = min(self.k, self.l)
        C = np.dot(self.Q.T, A)
        (Q_perp, B_perp) = np.linalg.qr(
            A - np.dot(self.Q, C))
        Q_hat = np.hstack([self.Q, Q_perp])
        B_hat = np.zeros((kt + lt, kt + lt))
        B_hat[:kt,:kt] += np.diag(self.B)
        B_hat[:kt,kt:] += C
        B_hat[kt:,:kt] += C.T
        B_hat[kt:,kt:] += B_perp

        return (Q_hat, B_hat)
