import numpy as np

from linal.utils import get_quadratic as gq
from drrobert.network import get_random_parameter_graph as get_rpg

class L2RegularizedRandomParameterGraph:

    def __init__(self, N, p, fc=True, dist=True, sym=True):
        # TODO: actually use the symmetric graph option

        self.N = N
        self.p = p

        (self.ws, self.Bw, self.Dw, self.G) = get_rpg(
            self.N, 
            self.p, 
            fc=fc,
            dist=dist)
        self.d = np.sum(self.G, axis=1)
        self.L = np.diag(d) - self.G

        (self.e_vals, self.e_vecs) = np.linalg.eig(self.L)

    def get_logdet(self, loc, scale):

        log_e_vals = np.log(scale * self.e_vals + loc)

        return np.sum(log_e_vals)

    def get_L_and_inv(self, loc, scale):

        aug_L_e_vals = scale * self.e_vals + loc
        lambda_matrix = np.diag(aug_L_e_vals)
        lambda_matrix_inv = np.diag(
            np.power(aug_L_e_vals, -1))

        aug_L = gq(self.e_vecs, lambda_matrix)
        aug_L_inv = gq(self.e_vecs, lambda_matrix_inv)

        return (aug_L, aug_L_inv)
