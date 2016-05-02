import numpy as np
import utils as agu

"""
Consider keeping a log of the minibatches somewhere for the online version.
"""

class AppGradCCA:

    def __init__(self,
        X_ds, Y_ds, k,
        online=False,
        eta1=0.1, eta2=0.1,
        eps1=10**(-3), eps2=10**(-3),
        ftprl1=None, ftprl2=None):

        if not agu.is_k_valid([X_ds, Y_ds], k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')
        else:
            self.k = k

        self.do_ftprl1 = ftprl1 is not None
        self.do_ftprl2 = ftprl2 is not None
        self.ftprl1 = ftprl1
        self.ftprl2 = ftprl2

        (self.X, self.Sx) = X_ds.get_batch_and_gram()
        (self.Y, self.Sy) = Y_ds.get_batch_and_gram()
        self.online = online
        
        if not online:
            # Find a better solution to this
            n = min([self.X.shape[0], self.Y.shape[0]])
            if self.X.shape[0] > n:
                self.X = self.X[:n,:]
            else:
                self.Y = self.Y[:n,:]

        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2

    def get_cca(self, verbose=False):

        (X, Sx, Y, Sy) = (self.X, self.Sx, self.Y, self.Sy)

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

            if self.online:
                # Update random minibatches if doing SGD
                (X, Sx) = self.X_ds.get_batch_and_gram()
                (Y, Sy) = self.Y_ds.get_batch_and_gram()

            if verbose:
                print "\tGetting updated basis estimates"

            # Get unconstrained, unnormalized gradients
            unn_Phi_grad = agu.get_gradient(
                X, unn_Phi_t, np.dot(Y, Psi_t))
            unn_Psi_grad = agu.get_gradient(
                Y, unn_Psi_t, np.dot(X, Phi_t))

            if self.do_ftprl1:
                # Get (composite with l1 reg) mirror descent updates
                unn_Phi_t1 = self.ftprl1.get_implicit_update(
                        unn_Phi_t, unn_Phi_grad, eta1)
            else:
                # Make normal gradient updates
                unn_Phi_t1 = unn_Phi_t - eta1 * unn_Phi_grad

            if self.do_ftprl2:
                unn_Psi_t1 = self.ftprl2.get_implicit_update(
                        unn_Psi_t, unn_Psi_grad, eta2)
            else:
                unn_Psi_t1 = unn_Psi_t - eta2 * unn_Psi_grad

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

        return (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t)
