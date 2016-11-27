import numpy as np
import optimization.utils as ou

from optimization.stepsize import FixedScheduler as FXS

class DANE:

    def __init__(self,
        server,
        get_gradient,
        num_rounds=5,
        num_processes=1,
        eta_scheduler=None,
        mu_scheduler=None):

        self.server = server
        self.get_gradient = get_gradient
        self.num_rounds = num_rounds
        self.num_processes = num_processes

        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        if mu_scheduler is None:
            mu_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler
        self.mu_scheduler = mu_scheduler
        self.data = self.server.get_data()

    def get_parameters(self, init_parameters=None):

        wi = init_parameters

        for i in xrange(self.num_rounds):
            print 'poop'
