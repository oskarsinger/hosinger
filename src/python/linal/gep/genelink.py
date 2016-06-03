import numpy as np

from linal import get_q
from linal.utils import multi_dot
from linal.utils import quadratic as quad
from svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

class GenELinK:

    def __init__(self, 
        epsilon=10**(-5)):

        self.epsilon = epsilon

    def fit(self, 
        A, B, k, 
        eta=0.1,
        optimizer=None,
        max_iter=1000, 
        verbose=False):

        (nA, pA) = A.shape
        (nB, pB) = B.shape
        ds = [nA, pA, nB, pB]

        if not all([d == nA for d in ds]):
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        if optimizer is None:
            optimizer = MAG()

        d = nA
        inner_prod = lambda x,y: multi_dot([x, B, y])

        # Initialize iteration variables
        W_t = get_q(np.random.randn(d, k), inner_prod=inner_prod)
        W_t1 = None
        i = 0

        while not converged and i < max_iter:

            # Compute initialization for trace minimization
            B_term = get_svd_power(quad(W, B), power=-1)
            A_term = quad(W, A)
            init = multi_dot([W_t, B_term, A_term])

            # Get (t+1)-th iterate of W
            unn_W_t1 = self._get_new_W(A, B, init, verbose)
            W_t1 = get_q(unn_W_t1, inner_prod=inner_prod)

            # Check for convergence
            converged = is_converged(
                W_t, W_t1, self.epsilon, verbose)

            # Update iteration variables
            W_t = np.copy(W_t1)
            i += 1

        return W_t1

    def _get_new_W(self, A, B, init, eta, optimizer, verbose):

        # Initialize iteration variables
        t = init 
        t1 = None
        i = 1

        while not converged:
            # Update iteration variable
            eta_t = eta / i**(0.5)

            # Get new parameter estimate
            gradient = np.dot((0.5*B - A).T, t)
            t1 = optimizer.get_update(t, gradient, eta_t)

            # Check for convergence
            converged = is_converged(t, t1, self.epsilon, verbose)

            # Update iteration variables
            t = np.copy(t1)

        return np.copy(t1)

    def _get_objective(self, A, B, W):

        inner = quad(W, 0.5*B - A)

        return np.trace(inner)
