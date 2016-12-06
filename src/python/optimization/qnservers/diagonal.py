import numpy as np
import optimization.utils as ou

from linal.utils import get_safe_power as get_sp

class StaticDiagonalServer:

    def __init__(self, D):

        self.D = D

        self.D_inv = get_sp(self.D, -1)
        self.lam = np.min(self.D)
        self.L = np.max(self.D)
        self.num_rounds = 0

    def get_qn_transform(self, search_direction):

        self.num_rounds += 1

        return self.D_inv * search_direction

    def get_qn_matrix(self):

        return np.diag(self.D)

    def get_qn_inverse(self):

        return np.diag(self.D_inv) 

    def get_lambda(self):

        return self.lam

    def get_L(self):

        return self.L
