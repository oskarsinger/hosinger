import numpy as np
import drrobert.debug as drdb

from .. import utils as ou
from optimization.utils import get_shrunk_and_thresholded as get_st
from linal.utils import get_safe_power
from linal.svd_funcs import get_multiplied_svd, get_svd_power
from drrobert.arithmetic import get_moving_avg as get_ma

class DiagonalAdamOptimizer:

    def __init__(self, 
        delta=1,
        beta1=None,
        beta2=None,
        lower=None, 
        dual_avg=True, 
        verbose=False):

        # TODO: try to enforce correct step-size sequence for RDA
        self.lower = lower
        self.dual_avg = dual_avg
        self.delta = delta
        self.verbose = verbose

        if dual_avg and (beta1 is not None or beta2 is not None):
            raise ValueError(
                'You must choose either dual averaging or moving average for the initial search direction.')
        
        if beta1 is None:
            beta1 = 0
            self.alpha1 = 1
        else:
            self.alpha1 = 1 - beta1

        self.beta1 = beta1

        if beta2 is None:
            beta2 = 0
            self.alpha2 = 1
        else:
            self.alpha2 = 1 - beta2

        self.beta2 = beta2

        self.first_moment = None
        self.second_moment = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        drdb.check_for_large_numbers(
            parameters,
            'DADO get_update at round ' + str(self.num_rounds),
            'parameters')
        drdb.check_for_large_numbers(
            gradient,
            'DADO get_update at round ' + str(self.num_rounds),
            'gradient')

        #print 'Inside DADO computing new'
        new = get_safe_power(gradient, 2)

        drdb.check_for_large_numbers(
            new,
            'DADO get_update first else body at round ' + str(self.num_rounds),
            'new')
        #print 'Inside DADO computing total'
        denom = (1 - self.beta2**(self.num_rounds))

        drdb.check_for_small_numbers(
            np.array([denom]),
            'DADO get_update first else body at round ' + str(self.num_rounds),
            'denom',
            exponent=-2)

        unnormed_total = get_ma(
            self.second_moment, 
            new, 
            self.alpha2, 
            self.beta2)

        drdb.check_for_large_numbers(
            unnormed_total,
            'DADO get_update first else body at round ' + str(self.num_rounds),
            'unnormed_total')

        normed = lambda: unnormed_total / denom

        drdb.handle_runtime_warning(
            normed, 'Denom: ' + str(denom))

        self.second_moment = normed()

        drdb.check_for_large_numbers(
            self.second_moment,
            'DADO get_update first else body at round ' + str(self.num_rounds),
            'second_moment')
        #print 'Inside DADO updating search direction'
        # Update search direction
        denom = (1 - self.beta1**(self.num_rounds))

        drdb.check_for_small_numbers(
            np.array([denom]),
            'DADO get_update at round ' + str(self.num_rounds),
            'denom',
            exponent=-2)

        self.first_moment = np.copy(ou.get_avg_first_moment(
            self.first_moment, 
            gradient, 
            self.dual_avg, 
            self.num_rounds,
            alpha=self.alpha1,
            beta=self.beta1)) / denom

        drdb.check_for_large_numbers(
            self.first_moment,
            'DADO get_update at round ' + str(self.num_rounds),
            'first_moment')
        #print 'Inside DADO getting mirror update'
        mirror_update = ou.get_mirror_update(
            parameters, 
            eta, 
            self.first_moment, 
            self._get_dual, 
            self._get_primal)

        drdb.check_for_large_numbers(
            mirror_update,
            'DADO get_update at round ' + str(self.num_rounds),
            'mirror_update')
        #print 'Inside DADO returning mirror update'

        return mirror_update

    def _get_dual(self, parameters):

        drdb.check_for_large_numbers(
            parameters,
            'DADO _get_dual at round' + str(self.num_rounds), 
            'parameters')

        #print 'Inside DADO._get_dual computing H'
        # Get the dual transformation
        H = get_safe_power(self.second_moment + self.delta, 0.5)

        drdb.check_for_large_numbers(
            H,
            'DADO _get_dual at round' + str(self.num_rounds), 
            'H')
        #print 'Returning H * parameters'

        return H * parameters

    def _get_primal(self, dual_update):

        if self.lower is not None:
            dus = dual_update.shape

            #print 'Inside soft thresholding with dus', dus

            if len(dus) == 2 and not 1 in set(dus):
                #print 'Inside matrix soft thresholding'
                (U, s, V) = np.linalg.svd(dual_update)
                #print 'Thresholding singular values'
                sparse_s = get_st(s, lower=self.lower)
                #print 'Remultiplying SVD'
                dual_update = get_multiplied_svd(U, s, V)
            else:
                #print 'Inside vector soft thresholding'
                dual_update = get_st(
                    dual_update, lower=self.lower) 

        #print 'Computing transformation back to primal space'

        # Get the primal transformation
        H_inv = get_safe_power(self.second_moment + self.delta, -0.5)
            
        drdb.check_for_large_numbers(
            H_inv,
            'DADO _get_primal at round ' + str(self.num_rounds),
            'H_inv')

        #print 'Returning primal parameters'

        return H_inv * dual_update

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'second_moment': self.second_moment,
            'alpha1': self.alpha1,
            'beta1': self.beta1,
            'alpha2': self.alpha2,
            'beta2': self.beta2,
            'dual_avg': self.dual_avg,
            'grad': self.first_moment,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}

class FullAdamOptimizer:

    def __init__(self,
        delta=1,
        beta1=None,
        beta2=None,
        lower=None,
        dual_avg=True,
        verbose=False):

        self.lower = lower
        self.dual_avg = dual_avg
        self.delta = delta
        self.verbose = verbose

        if dual_avg and beta1 is not None:
            raise ValueError(
                'You must choose either dual averaging or moving average for the initial search direction.')
        
        if beta1 is None:
            beta1 = 0
            self.alpha1 = 1
        else:
            self.alpha1 = 1 - beta1

        self.beta1 = beta1

        if beta2 is None:
            beta2 = 0
            self.alpha2 = 1
        else:
            self.alpha2 = 1 - beta2

        self.beta2 = beta2
        self.first_moment = None
        self.second_moment = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        # Update step sizes
        if self.second_moment is None:
            self.second_moment = np.dot(gradient, gradient.T)
        else:
            new_second_moment = np.dot(gradient, gradient.T)
            denom = (1 - self.beta2**(self.num_rounds))
            self.second_moment = get_ma(
                self.second_moment, 
                new_second_moment, 
                self.alpha2, 
                self.beta2) / denom

        # Update gradient
        denom = (1 - self.beta1**(self.num_rounds))
        self.first_moment = np.copy(ou.get_avg_first_moment(
            self.first_moment, 
            gradient, 
            self.dual_avg, 
            self.num_rounds,
            alpha=self.alpha1,
            beta=self.beta1)) / denom
        
        return ou.get_mirror_update(
            parameters, 
            eta, 
            self.first_moment, 
            self._get_dual, 
            self._get_primal)

    def _get_dual(self, parameters):

        # Get the dual transformation
        pd_help = self.delta * np.identity(self.second_moment.shape[0])
        H = get_svd_power(self.second_moment + pd_help, 0.5)

        return np.dot(H, parameters)

    def _get_primal(self, dual_update):

        dus = dual_update.shape

        if self.lower is not None:
            if len(dus) == 2 and not 1 in set(dus):
                (U, s, V) = np.linalg.svd(dual_update)
                sparse_s = get_st(s, lower=self.lower)
                dual_update = get_multiplied_svd(U, s, V)
            else:
                dual_update = get_st(
                    dual_update, lower=self.lower) 

        # Get the primal transformation
        pd_help = self.delta * np.identity(self.second_moment.shape[0])
        H_inv = get_svd_power(self.second_moment + pd_help, -0.5)

        return np.dot(H_inv, dual_update)

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'second_moment': self.second_moment,
            'alpha1': self.alpha1,
            'beta1': self.beta1,
            'alpha2': self.alpha2,
            'beta2': self.beta2,
            'dual_avg': self.dual_avg,
            'first_moment': self.first_moment,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
