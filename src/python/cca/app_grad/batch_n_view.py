import numpy as np

import utils as agu

class BatchAppGradNViewCCA:

    def __init__(self,
        ds_list, k,
        etas=None,
        epsilons=None,
        min_r=0.1,
        comids=None):

        self.num_ds = len(ds_list)

        if self.num_ds < 2:
            raise ValueError(
                'You must provide at least 2 data servers.')

        self.ds_list = ds_list

        if not agu.is_k_valid(ds_list, k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns over all data servers.')
        else:
            self.k = k

        if etas is not None:
            if not len(etas) == self.num_ds:
                raise ValueError(
                    'Length of etas and ds_list must be the same.')
            else:
                self.etas = etas
        else:
            self.etas = [0.1] * self.num_ds

        if epsilons is not None:
            if not len(epsilons) == self.num_ds:
                raise ValueError(
                    'Length of epsilons and ds_list must be the same.')
            else:
                self.epsilons = epsilons
        else:
            self.epsilons = [10**(-4)] * self.num_ds

        if comids is not None:
            if not len(comids) == self.num_ds:
                raise ValueError(
                    'Length of comids and ds_list must be the same.')
            else:
                self.comids = comids
                self.do_comid = [comid is not None 
                                 for comid in comids]
        else:
            self.comids = [None] * self.num_ds

        if min_r < 0:
            raise ValueError(
                'min_r must be non-negative.')
        else:
            self.min_r

        self.num_updates = [0] * self.num_ds
        (self.Xs, self.Sxs) = self._get_batch_and_gram_lists()

    def get_cca(self, verbose=False):

        print "Getting intial_basis_estimates"

        # Initialization of optimization variables
        basis_pairs_t = agu.get_init_basis_pairs(Sxs, self.k)
        basis_pairs_t1 = None

        # Iteration variables
        converged = [False] * self.num_ds
        i = 1

        while not all(converged):

            # Update step sizes
            etas = [eta / i**0.5 for eta in self.etas]
            i = i + 1

            if verbose:
                print "Iteration:", i
                print "\t".join(["eta" + str(i) + " " + str(eta)
                                 for eta in etas])

            # Update Psi
            Psi = self._get_Psi(Xs, basis_pairs_t)

            if verbose:
                print "\tGetting updated basis estimates"

            # Get updated canonical bases
            basis_pairs_t1 = self._get_basis_updates(
                basis_pairs_t, Xs, Psi, etas)

            if verbose:
                Phis = [pair[1] for pair in basis_pairs_t1]
                print "\tObjective:", agu.get_objective(Xs, Phis, Psi)

            # Check for convergence
            converged = agu.is_converged(
                [(basis_pairs_t[i][0], basis_pairs_t1[i][0])s, 
                 for i in range(self._num_ds)],
                self.epsilons,
                verbose)

            # Update iterates
            basis_pairs_t = [(np.copy(unn_Phi), np.copy(Phi))
                             for unn_Phi, Phi in basis_pairs_t1]

        return (basis_pairs_t, Psi)

    def _get_basis_updates(self, basis_pairs, Xs, Psi, etas):

        # Get gradients
        gradients = [agu.get_gradient(
                        Xs[i], basis_pairs[i][0], Psi)
                     for i in range(self.num_ds)]
        updated_unn = []

        for i in range(self.num_ds):
            unn_t = basis_pairs[i][0]
            eta = etas[i]
            grad = gradients[i]
            unn_t1 = None

            if self.do_comid[i]:
                # Get (composite with l1 reg) mirror descent update
                unn_t1 = self.comids[i].get_comid_update(
                    unn_t, grad, eta)
            else:
                # Make normal gradient update
                unn_t1 = unn_t - eta * grad

            updated_unn.append(unn_t1)

        # Normalize
        updated_pairs = [(unn, agu.get_gram_normed(unn, Sx))
                         for unn, Sx in zip(updated_unn, Sxs)]

        return updated_pairs

    def _get_Psi(self, basis_pairs, Psi):

        X_dot_Phis = [np.dot(self.Xs[i], basis_pairs[i][1]
                      for i in range(self.num_ds)]
        residuals = [np.linalg.norm(X_dot_Phi - Psi)
                     for X_dot_Phi in X_dot_Phis]

        # Truncate small residuals to prevent numerical instability
        truncd_rs = [max(r, self.min_r)
                     for r in residuals]

        # Weight the sum to prevent noise from dominating the loss
        summands = [r**(-1) * X_dot_Phi
                    for r, X_dot_Phi in zip(truncd_rs, X_dot_Phis)]

        return sum(summands)

    def _get_batch_and_gram_lists(self):

        batch_and_gram_list = [ds.get_batch_and_gram()
                               for ds in self._ds_list]
        Xs = [X for (X, Sx) in batch_and_gram_list]
        Sxs = [Sx for (X, Sx) in batch_and_gram_list]

        return (Xs, Sxs)
