import numpy as np

import utils as agu

class OnlineAppGradNViewCCA:

    def __init__(self,
        ds_list, k,
        etas=None,
        eps_list=None):

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

        if eps_list is not None:
            if not len(eps_list) == self.num_ds:
                raise ValueError(
                    'Length of eps_list and ds_list must be the same.')
            else:
                self.eps_list = eps_list
        else:
            self.eps_list = [10**(-4)] * self.num_ds

        self.num_updates = [0] * self.num_ds

    def get_cca(self, verbose=False):

        print "Getting initial minibatches and Sx matrices"

        # Determine minibatches and grams
        batch_and_gram_list = [ds.get_batch_and_gram()
                               for ds in self._ds_list]
        Xs = [X for (X, Sx) in batch_and_gram_list]
        Sxs = [Sx for (X, Sx) in batch_and_gram_list]
        
        print "Getting intial_basis_estimates"

        basis_pairs_t = agu.get_init_basis_pairs(Sxs, self.k)
        basis_pairs_t1 = [(None, None) for i in range(self.num_ds)]

        converged = [False] * self.num_ds
        i = 1

        while not all(converged):

            etas = [eta / i**0.5 for eta in self.etas]
            i = i + 1
