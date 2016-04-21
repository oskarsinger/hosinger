import numpy as np
import utils as agu

class BatchAppGradCCA:

    def __init__(self,
        X_ds, Y_ds, k,
        eta1=0.1, eta2=0.1,
        eps1=10**(-3), eps2=10**(-3),
        comid1=None, comid2=None):
     
        if not agu.is_k_valid([X_ds, Y_ds], k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')
        else:
            self.k = k

        self.do_comid1 = comid1 is not None
        self.do_comid2 = comid2 is not None
        self.comid1 = comid1
        self.comid2 = comid2

        (self.X, self.Sx) = X_ds.get_batch_and_gram()
        (self.Y, self.Sy) = Y_ds.get_batch_and_gram()
        n = min([self.X.shape[0], self.Y.shape[0]])
        
        # Find a cleaner way to do this
        if self.X.shape[0] > n:
            self.X = self.X[:n,:]
        else:
            self.Y = self.Y[:n,:]

        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2

    def get_cca(self, verbose=False):

        print "Getting initial basis estimates"

        # Randomly initialize normalized and unnormalized canonical bases for
        # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
        basis_pairs = agu.get_init_basis_pairs([self.Sx, self.Sy], self.k)
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
                print "\tGetting updated basis estimates"

            # Get unconstrained, unnormalized gradients
            unn_Phi_grad = agu.get_gradient(
                self.X, unn_Phi_t, np.dot(self.Y, Psi_t))
            unn_Psi_grad = agu.get_gradient(
                self.Y, unn_Psi_t, np.dot(self.X, Phi_t))

            if self.do_comid1:
                # Get (composite with l1 reg) mirror descent updates
                unn_Phi_t1 = self.comid1.get_comid_update(
                        unn_Phi_t, unn_Phi_grad, eta1)
            else:
                # Make normal gradient updates
                unn_Phi_t1 = unn_Phi_t - eta1 * unn_Phi_grad

            if self.do_comid2:
                unn_Psi_t1 = self.comid2.get_comid_update(
                        unn_Psi_t, unn_Psi_grad, eta2)
            else:
                unn_Psi_t1 = unn_Psi_t - eta2 * unn_Psi_grad

            # Normalize updated bases
            Phi_t1 = agu.get_gram_normed(unn_Phi_t1, self.Sx)
            Psi_t1 = agu.get_gram_normed(unn_Psi_t1, self.Sy)

            if verbose:
                print "\tObjective:", agu.get_2way_objective(
                    self.X, Phi_t1, self.Y, Psi_t1)

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
