import numpy as np

from linal.utils import get_quadratic as gq
from drrobert.network import get_random_parameter

class AugmentedLaplacianRandomParameterGraph:

    def __init__(self,
        parameters,
        adj_matrix):

        self.parameters = parameters
        self.A = adj_matrix

        self.n = self.A.shape[0]
        self.p = self.parameters[0].shape[0]

        Bws = [np.linalg.norm(p)
               for p in self.parameters]
        Dws = [np.linalg.norm(p1 - p2)
               for (i, p1) in enumerate(self.parameters)
               for p2 in self.parameters[i+1:]]

        self.Bw = max(Bws)
        self.Dw = max(Dws)

        trans_dist = np.linalg.norm(
            self.A - self.A.T)

        self.symmetric = trans_dist == 0
        self.d = np.sum(self.A, axis=1)
        self.L = np.diag(d) - self.A 

        (self.e_vals, self.e_vecs) = np.linalg.eig(self.L)

    def get_logdet(self, loc, scale):

        log_e_vals = np.log(scale * self.e_vals + loc)

        return np.sum(log_e_vals)

    def get_aug_L_and_inv(self, loc, scale):

        aug_L_e_vals = scale * self.e_vals + loc
        lambda_matrix = np.diag(aug_L_e_vals)
        lambda_matrix_inv = np.diag(
            np.power(aug_L_e_vals, -1))

        aug_L = gq(self.e_vecs, lambda_matrix)
        aug_L_inv = gq(self.e_vecs, lambda_matrix_inv)

        return (aug_L, aug_L_inv)
