import numpy as np

import utils as agu

class NViewAppGradCCA:

    def __init__(self,
        ds_list, k,
        online=False
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
            if not len(etas) == self.num_ds + 1:
                raise ValueError(
                    'Length of etas and ds_list + 1 must be the same.')
            else:
                self.etas = etas
        else:
            self.etas = [0.1] * (self.num_ds + 1)

        if epsilons is not None:
            if not len(epsilons) == self.num_ds + 1:
                raise ValueError(
                    'Length of epsilons and ds_list + 1 must be the same.')
            else:
                self.epsilons = epsilons
        else:
            self.epsilons = [10**(-4)] * (self.num_ds + 1)

        if comids is not None:
            if not len(comids) == self.num_ds + 1:
                raise ValueError(
                    'Length of comids and ds_list + 1 must be the same.')
            else:
                self.comids = comids
                self.do_comid = [comid is not None 
                                 for comid in comids]
        else:
            self.comids = [None] * (self.num_ds + 1)

        if min_r < 0:
            raise ValueError(
                'min_r must be non-negative.')
        else:
            self.min_r = min_r

        self.num_updates = [0] * (self.num_ds + 1)
        (self.Xs, self.Sxs) = self._get_batch_and_gram_lists()
        self.online = online

        if not self.online
            # Find a better solution to this
            self.n = min([X.shape[0] for X in self.Xs])
            self.Xs = [X[:self.n,:] for X in self.Xs]

    def get_cca(self, verbose=False):

        (Xs, Sxs) = (self.Xs, self.Sxs)

        print "Getting intial_basis_estimates"

        # Initialization of optimization variables
        basis_pairs_t = agu.get_init_basis_pairs(self.Sxs, self.k)
        basis_pairs_t1 = None
        Psi = np.random.randn(self.n, self.k)

        # Iteration variables
        converged = [False] * self.num_ds
        i = 1

        print "Starting gradient descent"

        while not all(converged):

            # Update step sizes
            etas = [eta / i**0.5 for eta in self.etas]

            if verbose:
                print "Iteration:", i
                print "\t".join(["eta" + str(i) + " " + str(eta)
                                 for eta in etas])
                if self.online:
                    print "\tGetting updated minibatches and grams"

            if self.online:
                # Get new minibatches and Gram matrices
                (Xs, Sxs) = self._get_batch_and_gram_lists()

            if verbose:
                print "\tGetting updated basis estimates"

            # Get updated canonical bases
            basis_pairs_t1 = self._get_basis_updates(
                Xs, Sxs, basis_pairs_t, Psi, etas)

            if verbose:
                print "\tGetting updated auxiliary variable estimate"

            # Get updated auxiliary variable
            Psi = self._get_Psi_update(
                Xs, basis_pairs_t1, Psi, etas[-1])

            if verbose:
                unns_t = [pair[0] for pair in basis_pairs_t]
                unns_t1 = [pair[0] for pair in basis_pairs_t1]
                dists = [np.linalg.norm(unn_t - unn_t1)
                         for unn_t, unn_t1 in zip(unns_t, unns_t1)]
                dist_strings = [str(i) + ": " + str(dist)
                                for i, dist in enumerate(dists)]
                dist_string = "\t".join(dist_strings)

                print "\tDistance between unnormed Phi iterates:", dist_string
                
                Phis_t1 = [pair[1] for pair in basis_pairs_t1]

                print "\tObjective:", agu.get_objective(self.Xs, Phis_t1, Psi)

            # Check for convergence
            converged = agu.is_converged(
                [(basis_pairs_t[j][0], basis_pairs_t1[j][0])
                 for j in range(self.num_ds)],
                self.epsilons,
                verbose)

            # Update iterates
            basis_pairs_t = [(np.copy(unn_Phi), np.copy(Phi))
                             for unn_Phi, Phi in basis_pairs_t1]

            i += 1

        return (basis_pairs_t, Psi)

    def _get_basis_updates(self, Xs, Sxs, basis_pairs, Psi, etas):

        # Get gradients
        gradients = [agu.get_gradient(Xs[i], basis_pairs[i][0], Psi)
                     for i in range(self.num_ds)]

        # Get unnormalized updates
        updated_unn = [self._get_parameter_update(
                        basis_pairs[i][0], gradients[i], etas[i], i)
                       for i in range(self.num_ds)]

        # Normalize
        updated_pairs = [(unn, agu.get_gram_normed(unn, Sx))
                         for unn, Sx in zip(updated_unn, Sxs)]

        return updated_pairs

    def _get_Psi_update(self, Xs, basis_pairs, Psi, eta):

        Phis = [pair[1] for pair in basis_pairs]
        gradient = agu.get_Psi_gradient(Psi, Xs, Phis)

        return self._get_parameter_update(Psi, gradient, eta, -1)

    def _get_parameter_update(self, parameter, gradient, eta, i):

        if self.do_comid[i]:
            # Get (composite with l1 reg) mirror descent update
            parameter = self.comids[i].get_comid_update(
                parameter, gradient, eta)
        else:
            # Make normal gradient update
            parameter = parameter - eta * gradient

        return parameter

    def _get_batch_and_gram_lists(self):

        batch_and_gram_list = [ds.get_batch_and_gram()
                               for ds in self.ds_list]
        Xs = [X for (X, Sx) in batch_and_gram_list]
        Sxs = [Sx for (X, Sx) in batch_and_gram_list]

        return (Xs, Sxs)
