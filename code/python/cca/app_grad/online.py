import numpy as np

import utils as agu

class OnlineAppGradCCA:

    def __init__(self, 
        data_server, k,
        batch_size=None,
        eta1=0.1, eta2=0.1, 
        eps1=10**(-4), eps2=10**(-4), 
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

        if batch_size is not None and batch_size < k:
            raise ValueError(
                'Batch size must be greater than or equal to the value of k.')

        self.ds = data_server
        self.eta1 = eta1
        self.eta2 = eta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.reg = reg

    def get_cca(self, verbose=False):

        print "Getting initial minibatches and Sx and Sy"

        # Determine data set
        X = self.ds.get_minibatch(self.X)
        Y = self.ds.get_minibatch(self.Y)
        Sx = self.ds.get_gram(X, self.reg)
        Sy = self.ds.get_gram(Y, self.reg)

        print "Getting initial basis estimates"

        # Randomly initialize normalized and unnormalized canonical bases for
        # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
        (Phi_t, unn_Phi_t, Psi_t, unn_Psi_t) = agu.get_init_bases(Sx, Sy, self.k)
        (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

        # Initialize iteration-related variables
        converged = False
        i = 1

        while not converged:

            # Update step scales for gradient updates
            eta1 = self.eta1 / i**0.5
            eta2 = self.eta2 / i**0.5
            i = i + 1

            if verbose:
                print "Iteration:", i
                print "\teta1:", eta1, "\teta2:", eta2
                print "\tGetting updated minibatches and Sx and Sy"

            # Update random minibatches if doing SGD
            X = self.ds.get_minibatch(self.X)
            Y = self.ds.get_minibatch(self.Y)
            Sx = self.ds.get_gram(X, self.reg)
            Sy = self.ds.get_gram(Y, self.reg)

            
