import numpy as np
import optimization.utils as ou

from optimization.stepsize import FixedScheduler as FXS

class StochasticVarianceReducedGradient:

    def __init__(self, 
        server, 
        get_gradient,
        num_rounds=5,
        factor=5,
        eta_scheduler=None):

        self.server = server
        self.get_gradient = get_gradient
        self.outer_num_rounds = num_rounds
        self.inner_num_rounds = factor * self.server.rows()

        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler
        self.data = self.server.get_data()
    
    def get_parameters(self, init_parameters=None):

        wi = init_parameters

        for i in xrange(self.outer_num_rounds):
            grad_wi = self.get_gradient(self.data, wi)
            wj = wi

            for j in xrange(self.inner_num_rounds):
                data_point = ou.get_minibatch(
                    self.data, 1)
                s_grad_wi = self.get_gradient(data_point, wi)
                s_grad_wj = self.get_gradient(data_point, wj)
                search_direction = s_grad_wj - s_grad_wi + grad_wi
                eta = self.eta_scheduler.get_stepsize()
                wj = wj - eta * search_direction

            wi = np.copy(wj)

        return wi
