import numpy as np
import optimization.utils as ou

from optimization.stepsize import FixedScheduler as FXS
from multiprocessing import Pool

class DANE:

    def __init__(self,
        servers,
        get_gradient,
        num_rounds=5,
        eta_schedulers=None,
        mu_schedulers=None):

        self.servers = servers
        self.num_nodes = len(self.servers)
        self.get_gradient = get_gradient
        self.num_rounds = num_rounds

        if eta_schedulers is None:
            eta_schedulers = [FXS(0.1) 
                              for i in xrange(self.num_nodes)]

        if mu_scheduler is None:
            mu_scheduler = [FXS(0.1)
                            for i in xrange(self.num_nodes)]

        node_stuff = zip(
            servers,
            eta_schedulers,
            mu_schedulers)

        self.nodes = [DANENode(s, self.get_gradient, es, ms)
                      for i in xrange(self.num_nodes)]

    def get_parameters(self, init_parameters=None):

        w_t = init_parameters

        for t in xrange(self.num_rounds):
            grad_t = self._get_node_avg(
                _get_node_grad, [w_t])
            w_t = self._get_node_avg(
                _get_node_update, [w_t, grad_t])

        return w_t

    def _get_node_avg(self, func, params):

        pool = Pool(processes=self.num_nodes)
        get_ps = lambda n, ps: [n] + [np.copy(p) for p in ps]
        results = [pool.apply_async(func, get_ps(n, params))
                   for n in self.nodes]
        things = [r.get() for r in results]

        return sum(things) / self.num_nodes

class DANENode:

    def __init__(self,
        server,
        get_gradient,
        eta_scheduler=None,
        mu_scheduler=None):

        self.server = server
        self.get_gradient = get_gradient
        
        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        if mu_scheduler is None:
            mu_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler
        self.mu_scheduler = mu_scheduler
        self.data = self.server.get_data()

    def get_update(self, w, global_grad):

        mu = self.mu_scheduler.get_stepsize()
        eta = self.eta_scheduler.get_stepsize()
        grad = self.get_gradient(self.data, w)
        global_term = grad - eta * global_grad
        local_term = 'Something objective dependent, unless I linearize?'

        print 'Poop'

    def get_gradient(self, w):

        return self.get_gradient(self.data, w)

def _get_node_grad(node, w):

    return node.get_gradient(w)

def _get_node_update(node, w, grad):

    return node.get_update(w, grad)
