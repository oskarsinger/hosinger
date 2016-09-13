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

        self.scale = None

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

        self.search_direction = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        drdb.check_for_nan_or_inf(
            parameters, 'DADO get_update', 'parameters')
        drdb.check_for_nan_or_inf(
            gradient, 'DADO get_update', 'gradient')

        # Update step sizes
        if self.scale is None:
            #print 'Inside DADO initializing scale'
            self.scale = np.absolute(gradient)

            drdb.check_for_nan_or_inf(
                self.scale, 'DADO get_update first if body', 'scale')
        else:
            #print 'Inside DADO computing old'
            old = get_safe_power(self.scale, 2)

            drdb.check_for_nan_or_inf(
                old, 'DADO get_update first else body', 'old')

            #print 'Inside DADO computing new'
            new = get_safe_power(gradient, 2)

            drdb.check_for_nan_or_inf(
                new, 'DADO get_update first else body', 'new')

            #print 'Inside DADO computing total'
            denom = (1 - self.beta2**(self.num_rounds))
            unnormed_total = get_ma(
                old, 
                new, 
                self.alpha2, 
                self.beta2)
            normed = lambda: unnormed_total / denom
            drdb.handle_runtime_warning(
                normed, 'Denom: ' + str(denom))

            drdb.check_for_nan_or_inf(
                total, 'DADO get_update first else body', 'total')

            #print 'Inside DADO updating scale'
            self.scale = get_safe_power(total, 0.5)

            drdb.check_for_nan_or_inf(
                self.scale, 'DADO get_update first else body', 'scale')

        #print 'Inside DADO updating search direction'
        # Update search direction
        denom = (1 - self.beta1**(self.num_rounds))
        self.search_direction = np.copy(ou.get_avg_search_direction(
            self.search_direction, 
            gradient, 
            self.dual_avg, 
            self.num_rounds,
            alpha=self.alpha1,
            beta=self.beta1)) / denom

        drdb.check_for_nan_or_inf(
            self.search_direction, 'DADO get_update', 'search_direction')
        
        #print 'Inside DADO getting mirror update'
        mirror_update = ou.get_mirror_update(
            parameters, 
            eta, 
            self.search_direction, 
            self._get_dual, 
            self._get_primal)

        drdb.check_for_nan_or_inf(
            mirror_update, 'DADO get_update', 'mirror_update')

        #print 'Inside DADO returning mirror update'

        return mirror_update

    def _get_dual(self, parameters):

        drdb.check_for_nan_or_inf(
            parameters, 'DADO _get_dual', 'parameters')

        #print 'Inside DADO._get_dual computing H'
        # Get the dual transformation
        H = self.scale + self.delta

        drdb.check_for_nan_or_inf(
            H, 'DADO _get_dual', 'H')

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
        H_inv = get_safe_power(self.scale + self.delta, -1)
            
        drdb.check_for_nan_or_inf(
            H_inv, 'DADO _get_primal', 'H_inv')

        #print 'Returning primal parameters'

        return H_inv * dual_update

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'scale': self.scale,
            'alpha1': self.alpha1,
            'beta1': self.beta1,
            'alpha2': self.alpha2,
            'beta2': self.beta2,
            'dual_avg': self.dual_avg,
            'grad': self.search_direction,
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

        self.G = None
        self.scale = None

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
        self.search_direction = None
        self.num_rounds = 0

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        # Update step sizes
        if self.scale is None:
            self.G = np.dot(gradient, gradient.T)
        else:
            new_G = np.dot(gradient, gradient.T)
            denom = (1 - self.beta2**(self.num_rounds))
            self.G = get_ma(
                self.G, 
                new_G, 
                self.alpha2, 
                self.beta2) / denom

        self.scale = get_svd_power(self.G, 0.5)

        # Update gradient
        denom = (1 - self.beta1**(self.num_rounds))
        self.search_direction = np.copy(ou.get_avg_search_direction(
            self.search_direction, 
            gradient, 
            self.dual_avg, 
            self.num_rounds,
            alpha=self.alpha1,
            beta=self.beta1)) / denom
        
        return ou.get_mirror_update(
            parameters, 
            eta, 
            self.search_direction, 
            self._get_dual, 
            self._get_primal)

    def _get_dual(self, parameters):

        # Get the dual transformation
        pd_help = self.delta * np.identity(self.scale.shape[0])
        H = self.scale + pd_help

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
        pd_help = self.delta * np.identity(self.scale.shape[0])
        H_inv = get_svd_power(self.scale + pd_help, -1)

        return np.dot(H_inv, dual_update)

    def get_status(self):

        return {
            'delta': self.delta,
            'lower': self.lower,
            'G': self.G,
            'scale': self.scale,
            'alpha1': self.alpha1,
            'beta1': self.beta1,
            'alpha2': self.alpha2,
            'beta2': self.beta2,
            'dual_avg': self.dual_avg,
            'search_direction': self.search_direction,
            'verbose': self.verbose,
            'num_rounds': self.num_rounds}
