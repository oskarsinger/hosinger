import numpy as np

from numpy.random import randn, choice
from linal.utils import quadratic as quad, get_svd_invert

class AppGradCCA:

    def __init__(self, 
        X, Y, k,
        batch_size=None,
        eta1=0.1, eta2=0.1, 
        eps1=10**(-5), eps2=10**(-5), 
        reg=0.001):

        (n1, p1) = X.shape
        (n2, p2) = Y.shape
        p = min([p1, p2])
        
        if not n1 == n2:
            raise ValueError(
                'Data matrices must have same number of data points.')
        else:
            self.X = X
            self.Y = Y

        if k > p:
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')
        else:
            self.k = k

        if batch_size is not None:
            if batch_size < k:
                raise ValueError(
                    'Batch size must be greater than or equal to the value of k.')
            else:
                self.stochastic = True
                self.batch_size = batch_size
        else:
            self.stochastic = False

        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.reg = reg
    
    def get_cca(self):

        # Determine data set
        X = self._get_minibatch(self.X) if self.stochastic else self.X
        Y = self._get_minibatch(self.Y) if self.stochastic else self.Y

        """
        # Normalize features
        X_sum = np.sum(X, axis=0)
        X_sum[X_sum == 0] = 1
        Y_sum = np.sum(Y, axis=0)
        Y_sum[Y_sum == 0] = 1
        X = X / X_sum
        Y = Y / Y_sum
        """

        print "Getting Sx and Sy"

        Sx = self._get_regged_gram(X)
        Sy = self._get_regged_gram(Y)

        print (
            "Sx shape:", Sx.shape, 
            "Sx rank:", np.linalg.matrix_rank(Sx),
            "Sx cond:", np.linalg.cond(Sx))
        print (
            "Sy shape:", Sy.shape,
            "Sy rank:", np.linalg.matrix_rank(Sy),
            "Sy cond:", np.linalg.cond(Sy))

        print "Getting initial basis estimates"

        # Randomly initialize normalized and unnormalized canonical bases for 
        # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
        (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t) = self._get_init_bases(Sx, Sy)
        (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

        # Initialize iteration-related variables
        converged = False
        i = 1

        while not converged:

            print "Iteration:", i

            # Update step scales for gradient updates
            eta1 = self.eta1 / i**0.5
            eta2 = self.eta2 / i**0.5

            print "\teta1:", eta1, "\teta2:", eta2

            i = i + 1

            print "\tGetting updated basis estimates"

            # Update random minibatches if doing SGD
            if self.stochastic:
                X = self._get_minibatch(self.X)
                Y = self._get_minibatch(self.Y)
                Sx = self._get_regged_gram(X)#(self._get_regged_gram(X) + (i - 1) * Sx) / i
                Sy = self._get_regged_gram(Y)#(self._get_regged_gram(Y) + (i - 1) * Sy) / i

                print (
                    "Sx shape:", Sx.shape, 
                    "Sx rank:", np.linalg.matrix_rank(Sx),
                    "Sx cond:", np.linalg.cond(Sx))
                print (
                    "Sy shape:", Sy.shape,
                    "Sy rank:", np.linalg.matrix_rank(Sy),
                    "Sy cond:", np.linalg.cond(Sy))

            # Get basis updates for both X and Y's canonical bases, normed and unnormed
            (unn_Phi_t1, Phi_t1) = self._get_updated_bases(
                X, Y, unn_Phi_t, Psi_t, Sx, eta1)
            (unn_Psi_t1, Psi_t1) = self._get_updated_bases(
                Y, X, unn_Psi_t, Phi_t, Sy, eta2)

            print "\tPhi orthogonal?", np.linalg.norm(quad(Phi_t1, Sx) - np.identity(Phi_t1.shape[1]))
            print "\tPsi orthogonal?", np.linalg.norm(quad(Psi_t1, Sy) - np.identity(Psi_t1.shape[1]))

            print "\tChecking for convergence"

            # Calculate distance between current and previous iterates of unnormalized 
            # canonical bases
            unn_Phi_dist = np.linalg.norm(unn_Phi_t - unn_Phi_t1)
            unn_Psi_dist = np.linalg.norm(unn_Psi_t - unn_Psi_t1)

            if np.isnan(unn_Phi_dist) or np.isnan(unn_Psi_dist):
                break

            print "\tUnnormalized Phi distance: ", unn_Phi_dist
            print "\tUnnormalized Psi distance: ", unn_Psi_dist

            # Check if distances are below tolerance threshold
            converged = unn_Phi_dist < self.eps1 and unn_Psi_dist < self.eps2

            print "\tObjective: ", np.linalg.norm(np.dot(X, Phi_t1) - np.dot(Y, Psi_t1))

            # Update state
            (unn_Phi_t, Phi_t, unn_Psi_t, Psi_t) = (
                np.copy(unn_Phi_t1), 
                np.copy(Phi_t1), 
                np.copy(unn_Psi_t1), 
                np.copy(Psi_t1))

        return (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t)

    def _get_minibatch(self, A):

        indexes = choice(
            np.arange(A.shape[0]), replace=False, size=self.batch_size)

        return A[indexes,:]

    def _get_regged_gram(self, A):

        gram = np.dot(A.T, A)
        reg_matrix = self.reg * np.identity(A.shape[1])

        return (gram + reg_matrix) / A.shape[0]

    def _get_updated_bases(self, X1, X2, unnormed1, normed2, S1, eta1):

        # Calculate the gradient with respect to unnormed1
        gradient = np.dot(X1.T, (np.dot(X1, unnormed1) - np.dot(X2, normed2)) ) / X1.shape[0]

        # Take a gradient step on unnormed1
        unnormed1_next = unnormed1 - eta1 * gradient

        # Normalize unnormed 1 with inversion of matrix quadratic
        normed1 = self._get_mah_normed(unnormed1_next, S1)

        print "\tOrthogonal?", quad(normed1, S1)

        return (unnormed1_next, normed1)

    def _get_init_bases(self, Sx, Sy):

        # Initialize Gaussian matrices for unnormalized bases
        unn_Phi = randn(Sx.shape[0], self.k)
        unn_Psi = randn(Sy.shape[0], self.k)

        # Normalize for initial normalized bases
        Phi = self._get_mah_normed(unn_Phi, Sx)
        Psi = self._get_mah_normed(unn_Psi, Sy) 

        return (Phi, unn_Phi, Psi, unn_Psi)

    def _get_mah_normed(self, unnormed, S):

        basis_quad = quad(unnormed, S)
        normalizer = get_svd_invert(
            basis_quad, random=False, power=-0.5)

        return np.dot(unnormed, normalizer)
