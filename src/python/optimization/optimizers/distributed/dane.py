import numpy as np
import optimization.utils as ou

from optimization.stepsize import FixedScheduler as FXS
from optimization.qnservers import DiagonalAdaGradServer as DAGS
from multiprocessing import Pool

class QuasinewtonInexactDANE:

    def __init__(self,
        servers,
        get_gradient,
        num_rounds=5,
        eta_schedulers=None,
        init_params=None):

        self.servers = servers
        self.num_nodes = len(self.servers)
        self.get_gradient = get_gradient
        self.num_rounds = num_rounds
        self.init_params = init_params
        self.w = None

        if eta_schedulers is None:
            eta_schedulers = [FXS(0.1) 
                              for i in xrange(self.num_nodes)]

        node_stuff = zip(
            servers,
            eta_schedulers)

        self.nodes = [DANENode(s, self.get_gradient, eta_scheduler=es)
                      for (s, es) in node_stuff]

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.w)

    def compute_parameters(self):

        w_t = np.copy(self.init_params)

        for t in xrange(self.num_rounds):
            local_grads = [n.get_gradient(np.copy(w_t))
                           for n in self.nodes]
            grad_t = sum(local_grads) / self.num_nodes
            updates = [n.get_update(
                            np.copy(w_t), 
                            np.copy(grad_t))
                       for n in self.nodes]
            w_t = sum(updates) / self.num_nodes
            """
            grad_funcs = [_get_grad_func(n, np.copy(w_t))
                          for n in self.nodes]
            grad_t = self._get_node_avg(grad_funcs)
            update_funcs = [_get_update_func(
                                n, 
                                np.copy(w_t), 
                                np.copy(grad_t))
                            for n in self.nodes]
            w_t = self._get_node_avg(update_funcs)
            """

        self.w = w_t

    """
    def _get_node_avg(self, funcs):

        pool = Pool(processes=self.num_nodes)
        get_ps = lambda n, ps: [n] + [np.copy(p) for p in ps]
        results = [pool.apply_async(f, ())
                   for f in funcs]
        things = [r.get() for r in results]

        return sum(things) / self.num_nodes
    """

class DANENode:

    def __init__(self,
        server,
        get_gradient,
        eta_scheduler=None,
        mu=1):

        self.server = server
        self.mu = mu
        
        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler
        self.qn_server = DAGS(delta=self.mu)
        self.data = self.server.get_data()
        self.get_gradient = lambda w: get_gradient(self.data, w)

    def get_update(self, w, global_grad):

        eta = self.eta_scheduler.get_stepsize()
        grad = self.get_gradient(self.data, w)
        search_direction = self.mu * w + grad - eta * global_grad

        return self.qn_server.get_qn_transform(search_direction)

"""
def _get_grad_func(n, w):

    def _get_node_grad():

        return n.get_gradient(w)

    return _get_node_grad

def _get_update_func(n, w, gg):

    def _get_node_update():

        return n.get_update(w, gg)

    return _get_node_update
"""
