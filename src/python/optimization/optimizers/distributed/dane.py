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
                      for i in xrange(self.num_nodes)]
        self.get_node_grad = lambda n, w: n.get_gradient(w)
        self.get_node_update = lambda n, w, gg: n.get_update(w, gg)

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        else:
            return np.copy(self.w)

    def compute_parameters(self):

        w_t = np.copy(self.init_params)

        for t in xrange(self.num_rounds):
            grad_t = self._get_node_avg(
                self.get_node_grad, [w_t])
            w_t = self._get_node_avg(
                self.get_node_update, [w_t, grad_t])

        self.w = w_t

    def _get_node_avg(self, func, params):

        pool = Pool(processes=self.num_nodes)
        get_ps = lambda n, ps: [n] + [np.copy(p) for p in ps]
        results = [pool.apply_async(func, get_ps(n, params))
                   for n in self.nodes]
        things = [r.get() for r in results]

        return sum(things) / self.num_nodes

class DANENode:

    def __init__(self,
        data_server,
        get_gradient,
        qn_server=None,
        eta_scheduler=None,
        mu=1):

        self.server = server
        self.get_gradient = get_gradient
        self.mu = mu
        
        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        if qn_server is None:
            qn_server = DAGS(delta=self.mu)

        self.eta_scheduler = eta_scheduler
        self.qn_server = qn_server
        self.data = self.server.get_data()

    def get_update(self, w, global_grad):

        eta = self.eta_scheduler.get_stepsize()
        grad = self.get_gradient(self.data, w)
        search_direction = self.mu * w + grad - eta * global_grad

        return self.qn_server.get_qn_transform(search_direction)

    def get_gradient(self, w):

        return self.get_gradient(self.data, w)
