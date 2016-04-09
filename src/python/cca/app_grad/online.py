import numpy as np

import utils as agu

class OnlineAppGradCCA:

    def __init__(self, 
        X_ds, Y_ds, k,
        eta1=0.1, eta2=0.1, 
        eps1=10**(-4), eps2=10**(-4)):

        if not agu.is_k_valid([X_ds, Y_ds], k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')
        else:
            self.k = k

        self.X_ds = X_ds
        self.Y_ds = Y_ds
        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2

    def get_cca(self, verbose=False):

        print "Getting initial minibatches and Sx and Sy"

        # Determine minibatch
        (X, Sx) = self.X_ds.get_batch_and_gram()
        (Y, Sy) = self.Y_ds.get_batch_and_gram()

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
                print "\tGetting updated minibatches and Sx and Sy"

            # Update random minibatches if doing SGD
            (X, Sx) = self.X_ds.get_batch_and_gram()
            (Y, Sy) = self.Y_ds.get_batch_and_gram()

            if verbose:
                print "\tGetting updated basis estimates"

            # Get basis updates for both X and Y's canonical bases, normed and unnormed
            (unn_Phi_t1, Phi_t1) = agu.get_2way_basis_update(
                X, Y, unn_Phi_t, Psi_t, Sx, eta1)
            (unn_Psi_t1, Psi_t1) = agu.get_2way_basis_update(
                Y, X, unn_Psi_t, Phi_t, Sy, eta2)

            # Need to reconsider how this is evaluated.
            # Its a bit sketchy to just look at objective for current batch.
            if verbose:
                print "\tObjective:", agu.get_2way_objective(
                    X, Phi_t1, Y, Psi_t1)

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

        return (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t)
