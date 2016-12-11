import numpy as np

from optimization.stepsize import FixedScheduler as FXS
from optimization.qnservers import DiagonalAdaGradServer as DAGS
from drrobert.misc import unzip

class QuasinewtonInexactDANE:

    def __init__(self,
        servers,
        get_gradient,
        get_objective,
        num_rounds=50,
        mu=100,
        eta_schedulers=None,
        init_params=None):

        self.servers = servers
        self.num_nodes = len(self.servers)
        self.get_gradient = get_gradient
        self.get_objective = get_objective
        self.num_rounds = num_rounds
        self.mu = mu
        self.init_params = init_params
        self.w = None

        if eta_schedulers is None:
            eta_schedulers = [FXS(0.1) 
                              for i in xrange(self.num_nodes)]

        node_stuff = zip(
            servers,
            eta_schedulers)

        self.nodes = [DANENode(
                        ds, 
                        self.get_gradient, 
                        self.get_objective,
                        eta_scheduler=es,
                        mu=self.mu)
                      for (ds, es) in node_stuff]
        self.objectives = []

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        
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
            (ws, objectives) = unzip(updates)
            # TODO: Make sure this is a good place to project
            w_t = sum(ws) / self.num_nodes

            self.objectives.append(sum(objectives))

        self.w = w_t

class DANENode:

    def __init__(self,
        server,
        get_gradient,
        get_objective,
        eta_scheduler=None,
        mu=100):

        self.server = server
        self.mu = mu
        
        if eta_scheduler is None:
            eta_scheduler = FXS(0.1)

        self.eta_scheduler = eta_scheduler
        self.qn_server = DAGS(delta=self.mu)
        self.data = self.server.get_data()
        self.get_gradient = lambda w: get_gradient(self.data, w)
        self.get_objective = lambda w: get_objective(self.data, w)

    def get_update(self, global_w, global_grad):

        eta = self.eta_scheduler.get_stepsize()
        search_direction = self.qn_server.get_qn_transform(
            global_grad)
        new_w = global_w - eta * search_direction
        objective = self.get_objective(new_w)

        return (new_w, objective)
