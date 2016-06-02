import numpy as np

from linal import get_q
from linal.utils import multi_dot

class GenELinK:

    def __init__(self):
    
        print "Stuff"

    def fit(self, A, B, k, max_iter):

        (nA, pA) = A.shape
        (nB, pB) = B.shape

        # Remember to assert that those are all the same
        d = nA
        inner_prod = lambda x,y: multi_dot([x, B, y])
        W = np.random.randn(A.shape[0], k)
