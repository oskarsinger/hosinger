import numpy as np

import utils as agu

class NViewAppGradCCA:

    def __init__(self,
        k, num_views,
        online=False,
        etas=None,
        epsilons=None,
        min_r=0.1):

        self.k = k

        if num_views < 2:
            raise ValueError(
                'You must provide at least 2 data servers.')
        else:
            self.num_views = num_views

        if etas is not None:
            self.etas = etas
        else:
            self.etas = [0.1] * (self.num_views + 1)

        if epsilons is not None:
            self.epsilons = epsilons
        else:
            self.epsilons = [10**(-4)] * (self.num_views + 1)

        if min_r < 0:
            raise ValueError(
                'min_r must be non-negative.')
        else:
            self.min_r = min_r

        self.num_updates = [0] * (self.num_views + 1)
        self.online = online
        self.has_been_fit = False
        self.basis_pairs = None
        self.Psi = None

    def fit(self,
        ds_list, 
        optimizers=None,
        verbose=False):

        if optimizers is not None:
            if not len(optimizers) == self.num_views + 1:
                raise ValueError(
                    'Length of optimizers and ds_list + 1 must be the same.')
        else:
            optimizers = [GradientOptimizer() 
                          for i in range(self.num_views + 1)]

        (Xs, Sxs) = self._init_data(ds_list)

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
                (Xs, Sxs) = self._get_batch_and_gram_lists(ds_list)

            if verbose:
                print "\tGetting updated basis estimates"

            # Get updated canonical bases
            basis_pairs_t1 = self._get_basis_updates(
                Xs, Sxs, basis_pairs_t, Psi, etas, optimizers)

            if verbose:
                print "\tGetting updated auxiliary variable estimate"

            # Get updated auxiliary variable
            Psi = self._get_Psi_update(
                Xs, basis_pairs_t1, Psi, etas[-1], optimizers[-1])

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

                print "\tObjective:", agu.get_objective(Xs, Phis_t1, Psi)

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

        self.has_been_fit = True
        self.basis_pairs = basis_pairs_t
        self.Psi = Psi

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return (self.basis_pairs, self.Psi)

    def _get_basis_updates(self, Xs, Sxs, basis_pairs, Psi, etas, optimizers):

        # Get gradients
        gradients = [agu.get_gradient(Xs[i], basis_pairs[i][0], Psi)
                     for i in range(self.num_views)]

        # Get unnormalized updates
        updated_unn = [optimizers[i].get_update(
                        basis_pairs[i][0], gradients[i], etas[i])
                       for i in range(self.num_views)]

        # Normalize with gram-parameterized Mahalanobis
        normed_pairs = [(unn, agu.get_gram_normed(unn, Sx))
                        for unn, Sx in zip(updated_unn, Sxs)]

        return updated_pairs

    def _get_Psi_update(self, Xs, basis_pairs, Psi, eta, optimizer):

        Phis = [pair[1] for pair in basis_pairs]
        gradient = self._get_Psi_gradient(Psi, Xs, Phis)

        return optimizer.get_update(Psi, gradient, eta, -1)

    def _get_batch_and_gram_lists(self, ds_list):

        batch_and_gram_list = [ds.get_batch_and_gram()
                               for ds in ds_list]
        Xs = [X for (X, Sx) in batch_and_gram_list]
        Sxs = [Sx for (X, Sx) in batch_and_gram_list]

        return (Xs, Sxs)

    def _get_Psi_gradient(self, Psi, Xs, Phis):

        summands = [np.dot(X, Phi)
                    for (X, Phi) in zip(Xs, Phis)]

        return (n * Phi - 2 * sum(summands)) / Psi.shape[0]

    def _init_data(self, ds_list):

        if not len(ds_list) == self.num_views:
            raise ValueError(
                'Parameter ds_list must have length num_views.')

        if not agu.is_k_valid(ds_list, self.k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        (Xs, Sxs) = self._get_batch_and_gram_lists(ds_list)

        if not self.online:
            # Find a better solution to this
            n = min([X.shape[0] for X in Xs])

            # Remove to-be-truncated examples from Gram matrices
            removed = [X[n:,:] for X in Xs]
            Sxs = [Sx - np.dot(r.T, r) 
                   for (Sx, r) in zip(Sxs, removed)]

            # Truncate extra examples
            Xs = [X[:n,:] for X in Xs]

        return (Xs, Sxs)
