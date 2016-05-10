import numpy as np
import utils as agu

from optimization.optimizers import GradientOptimizer

class AppGradCCA:

    def __init__(self,
        k=1,
        online=False,
        eta1=0.1, eta2=0.1,
        eps1=10**(-3), eps2=10**(-3)):

        self.k = k
        self.online = online
        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2

        self.has_been_fit = False
        self.Phi = None
        self.unn_Phi = None
        self.Psi = None
        self.unn_Psi = None

    def fit(self, 
        X_ds, Y_ds, 
        optimizer1=GradientOptimizer(), 
        optimizer2=GradientOptimizer(),
        verbose=False):

        (X, Sx, Y, Sy) = self._init_data(X_ds, Y_ds)

        print "Getting initial basis estimates"

        # Randomly initialize normalized and unnormalized canonical bases for
        # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
        basis_pairs = agu.get_init_basis_pairs([Sx, Sy], self.k)
        (Phi_t, unn_Phi_t) = basis_pairs[0]
        (Psi_t, unn_Psi_t) = basis_pairs[1]
        (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

        # Initialize iteration-related variables
        converged = [False] * 2
        i = 1

        while not all(converged):

            # Update step scales for gradient updates
            eta1 = self.eta1 / i**0.5
            eta2 = self.eta2 / i**0.5
            i = i + 1

            if verbose:
                print "Iteration:", i
                print "\teta1:", eta1, "\teta2:", eta2

                if self.online:
                    print "\tGetting updated minibatches and Sx and Sy"

            # Update random minibatches if doing SGD
            if self.online:
                (X, Sx) = self.X_ds.get_batch_and_gram()
                (Y, Sy) = self.Y_ds.get_batch_and_gram()

            if verbose:
                print "\tGetting updated basis estimates"

            # Get unconstrained, unnormalized gradients
            unn_Phi_grad = agu.get_gradient(
                X, unn_Phi_t, np.dot(Y, Psi_t))
            unn_Psi_grad = agu.get_gradient(
                Y, unn_Psi_t, np.dot(X, Phi_t))

            # Make updates to basis parameters
            unn_Phi_t1 = optimizer1.get_update(
                    unn_Phi_t, unn_Phi_grad, eta1)
            unn_Psi_t1 = optimizer2.get_update(
                    unn_Psi_t, unn_Psi_grad, eta2)

            # Normalize updated bases
            Phi_t1 = agu.get_gram_normed(unn_Phi_t1, Sx)
            Psi_t1 = agu.get_gram_normed(unn_Psi_t1, Sy)

            if verbose:
                Phi_dist = np.linalg.norm(unn_Phi_t1 - unn_Phi_t)
                Psi_dist = np.linalg.norm(unn_Psi_t1 - unn_Psi_t)
                print "\tDistance between unnormed Phi iterates:", Phi_dist
                print "\tDistance between unnormed Psi iterates:", Psi_dist
                print "\tObjective:", agu.get_2way_objective(
                    X, Phi_t1, Y, Psi_t1)

            # Check for convergence
            converged = agu.is_converged(
                [(unn_Phi_t, unn_Phi_t1), (unn_Psi_t, unn_Psi_t1)],
                [self.eps1, self.eps2],
                verbose)

            # Update state
            (unn_Phi_t, Phi_t, unn_Psi_t, Psi_t) = (
                np.copy(unn_Phi_t1),
                np.copy(Phi_t1),
                np.copy(unn_Psi_t1),
                np.copy(Psi_t1))

        print "Completed in", str(i), "iterations."

        self.has_been_fit = True
        self.Phi = Phi_t
        self.unn_Phi = unn_Phi_t
        self.Psi = Psi_t
        self.unn_Psi = unn_Psi_t

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return (self.Phi, self.Psi, self.unn_Phi, self.unn_Psi)

    def _init_data(self, X_ds, Y_ds):

        if not agu.is_k_valid([X_ds, Y_ds], self.k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        (X, Sx) = X_ds.get_batch_and_gram()
        (Y, Sy) = Y_ds.get_batch_and_gram()

        if not self.online:
            # Find a better solution to this
            n = min([X.shape[0], Y.shape[0]])
            if X.shape[0] > n:
                # Remove to-be-truncated examples from Gram matrix
                removed = X[n:,:]
                Sx -= np.dot(removed.T, removed)

                # Truncate extra examples
                X = X[:n,:]
            else:
                # Do the same for Y if Y has the extra examples
                removed = Y[n:,:]
                Sy -= np.dot(removed.T, removed)
                Y = Y[:n,:]

        return (X, Sx, Y, Sy)

