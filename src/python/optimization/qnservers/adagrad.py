import numpy as np
import optimization.utils as ou

from linal.utils import get_safe_power as get_sp
from drrobert.arithmetic import get_running_avg

class FullAdaGradServer:

    def __init__(self):

        print 'Poop'

# TODO: put in soft thresholding option for L1 penalty
class DiagonalAdaGradServer:

    def __init__(self,
        delta=1,
        verbose=False):

        self.delta = delta

        self.num_rounds = 0
        self.Q = None

    def get_qn_transform(self, search_direction):
        
        self.num_rounds += 1

        if self.Q is None:
            self.Q = np.abs(search_direction) 
        else:
            old = get_sp(self.Q, 2)
            new = get_sp(search_direction, 2)
            avg = get_running_avg(
                old, new, self.num_rounds)

            self.Q = get_sp(avg, 0.5)

        return get_sp(self.Q + self.delta, -1) * search_direction

    def get_qn_matrix(self):

        return np.diag(self.Q + self.delta)

    def get_qn_inverse(self):

        return np.diag(get_sp(self.Q + self.delta, -1))

    def get_lambda(self):

        return np.min(self.Q)

    def get_L(self):

        return np.max(self.Q)
