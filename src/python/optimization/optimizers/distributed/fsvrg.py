import numpy as np

from drrobert.misc import unzip, prod
from optimization.qnservers import StaticDiagonalServer as SDS
from optimization.stepsize import InversePowerScheduler as IPS
from linal.utils import get_safe_power as get_sp

class BanditFSVRG:

    def __init__(self,
        get_model,
        servers,
        max_rounds=10,
        h=0.01,
        init_params=None):

        self.get_model = get_model
        self.servers = servers
        self.num_nodes = len(self.servers)
        self.max_rounds = max_rounds
        self.h = h
        self.init_params = init_params

        self.w = None
        self.nodes = [BanditFSVRGNode(
                        self.get_model(i),
                        ds,
                        i,
                        h=self.h)
                      for (i, ds) in enumerate(self.servers)]
        self.objectives = [[] for i in xrange(self.num_nodes)]
        self.num_rounds = 0

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        
        return np.copy(self.w)

    def compute_parameters(self):

        w_t = np.copy(self.init_params)

        for i in xrange(self.max_rounds):
            grad_t = np.zeros_like(w_t)

            if i > 0:
                local_grads = [n.get_gradient(n.get_local(w_t))
                               for n in self.nodes]
                grad_t = np.vstack(local_grads) / self.num_nodes

            (S_servers, A_server) = self._get_S_and_A_servers()

            for (S_s, n) in zip(S_servers, self.nodes):
                n.set_global_info(w_t, grad_t)
                n.set_S_server(S_s)

            for j in xrange(2**i):
                self.num_rounds += 1

                for n in self.nodes:
                    n.set_local_action()
                    
                for n in self.nodes:
                    n.compute_local_update()

            updates = [n.get_update()
                       for n in self.nodes]
            (ws, objectives) = unzip(updates)
            agg = self._get_aggregate_grad(
                ws, w_t, A_server)
            w_t = w_t + agg

            for (i, o) in enumerate(objectives):
                self.objectives[i].append(o)

        #for n in self.nodes:
        #    print n.objectives

        self.w = w_t

    def _get_aggregate_grad(self, ws, w_t, A_server):

        weighted = [(float(nk) / self.n) * (w_k - w_t)
                    for (nk, w_k) in zip(self.nks, ws)]

        return A_server.get_qn_transform(
            sum(weighted))

    def _get_S_and_A_servers(self):

        (njks, phi_jks, self.nks) = unzip(
            [n.get_fsvrg_params()
             for n in self.nodes])
        self.n = sum(self.nks)
        njs = sum(njks)
        phi_js = njs / self.n
        sjks = [(phi_js / phi_jk)
                for phi_jk in phi_jks]
        S_servers = [SDS(sjk) for sjk in sjks]
        omega_js = np.vstack(
            (njk != 0).astype(float) 
            for njk in njks)
        ajs = get_sp(omega_js / self.num_nodes, -1)
        A_server = SDS(ajs)

        return (S_servers, A_server)

class BanditFSVRGNode:

    def __init__(self,
        model,
        server,
        id_number,
        h=0.01):

        self.model = model
        self.server = server
        self.get_objective = model.get_objective
        self.get_coord_counts = model.get_coordinate_counts
        self.get_action = model.get_action
        self.id_number = id_number
        self.h = h

        self.nk = 1
        self.p_shape = self.model.get_parameter_shape()
        # TODO: replace begin/end with arbitrary index vector
        self.begin = self.id_number * prod(self.p_shape)
        self.end = self.begin + prod(self.p_shape)
        self.get_local = lambda x: x[self.begin:self.end]
        (self.n_jks, self.phi_jks) = [None] * 2
        self.get_stochastic_gradient = model.get_gradient
        self.eta_scheduler = IPS(initial=h,power=0.75)
        self.objectives = []
        self.rewards = []
        self.actions = []

    def set_S_server(self, S_server):

        self.S_server = S_server

    def get_fsvrg_params(self):

        self._compute_local_fsvrg_params()

        return (self.n_jks, self.phi_jks, self.nk)

    def _compute_local_fsvrg_params(self):

        if self.nk == 1:
            self.n_jks = np.zeros(
                self.model.get_parameter_shape())
            # To avoid NaNs in S matrix
            self.n_jks += 0.0001
        else:
            self.n_jks = self.get_coord_counts(
                zip(self.rewards, self.actions))

        self.phi_jks = self.n_jks / self.nk

    def set_global_info(self, global_w, global_grad):

        self.global_w = np.copy(global_w)
        self.global_grad = np.copy(global_grad)
        self.local_w = self.get_local(global_w)
        self.local_grad = self.get_local(global_grad)
        self.w_n = np.copy(self.local_w)

    def get_update(self):

        return (self.global_w, self.objectives[-1])

    def set_local_action(self):

        action = self.get_action(self.global_w)

        self.actions.append(action)
        self.server.set_action(action)

    def compute_local_update(self):

        reward = self.server.get_reward()

        self.rewards.append(reward)

        datum = zip([reward], [self.actions[-1]])
        local_grad = self.get_stochastic_gradient(
            datum, self.local_w)
        grad_n = self.get_stochastic_gradient(
            datum, self.w_n)
        search_direction = self.S_server.get_qn_transform(
            grad_n - local_grad) + self.local_grad
        eta = self.eta_scheduler.get_stepsize()
        self.w_n -= eta * search_direction
        self.local_w = self.w_n
        self.nk += 1

        self.objectives.append(
            self.get_objective(
                zip(self.rewards, self.actions), 
                self.w_n) / self.nk)


    def get_gradient(self, w):

        data = zip(self.rewards, self.actions)

        return self.model.get_gradient(data, w)

class FSVRG:

    def __init__(self,
        get_model,
        servers,
        max_rounds=10,
        h=0.01,
        init_params=None):

        self.get_model = get_model
        self.servers = servers
        self.num_nodes = len(self.servers)
        self.max_rounds = max_rounds
        self.h = h
        self.init_params = init_params

        self.w = None
        self.nodes = [FSVRGNode(
                        self.get_model(i),
                        ds,
                        i,
                        h=self.h)
                      for (i, ds) in enumerate(self.servers)]
        self.S_servers = None
        self.A_server = None

        self._compute_S_and_A_servers()

        self.objectives = []

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        
        return np.copy(self.w)

    def compute_parameters(self):

        w_t = np.copy(self.init_params)

        for i in xrange(self.max_rounds):
            local_grads = [n.get_gradient(n.get_local(w_t))
                           for n in self.nodes]
            grad_t = np.vstack(local_grads) / self.num_nodes

            updates = [n.get_update(
                            np.copy(w_t),
                            np.copy(grad_t))
                       for n in self.nodes]
            (ws, objectives) = unzip(updates)
            agg = self._get_aggregate_grad(ws, w_t)
            w_t = w_t + agg

            self.objectives.append(
                [o[-1] for o in objectives])

        self.w = w_t

    def _get_aggregate_grad(self, ws, w_t):

        weighted = [(float(nk) / self.n) * (w_k - w_t)
             for (nk, w_k) in zip(self.nks, ws)]

        return self.A_server.get_qn_transform(
            sum(weighted))

    def _compute_S_and_A_servers(self):

        (njks, phi_jks, self.nks) = unzip(
            [n.get_fsvrg_params()
             for n in self.nodes])
        self.n = sum(self.nks)
        njs = sum(njks)
        phi_js = njs / self.n
        sjks = [(phi_js / phi_jk)[:,np.newaxis]
                    for phi_jk in phi_jks]
        self.S_servers = [SDS(sjk) for sjk in sjks]

        for (n, S_s) in zip(self.nodes, self.S_servers):
            n.set_S_server(S_s)

        omega_js = np.vstack(
            (njk[:,np.newaxis] != 0).astype(float) 
            for njk in njks)
        ajs = get_sp(omega_js / self.num_nodes, -1)

        self.A_server = SDS(ajs)

class FSVRGNode:

    def __init__(self,
        model,
        server,
        id_number,
        h=0.1):

        self.model = model
        self.server = server
        self.get_objective = model.get_objective
        self.get_coord_counts = model.get_coordinate_counts
        self.id_number = id_number

        self.nk = self.server.rows()
        self.p = self.server.cols()
        # TODO: replace begin/end with arbitrary index vector
        self.begin = self.id_number * self.p
        self.end = self.begin + self.p
        self.get_local = lambda x: x[self.begin:self.end]
        self.data = self.server.get_data()
        (self.n_jks, self.phi_jks) = [None] * 2

        self._compute_local_fsvrg_params()

        self.get_full_gradient = lambda w: model.get_gradient(
            self.data, w)
        self.get_stochastic_gradient = model.get_gradient
        self.eta = h / self.nk
        self.objectives = []

    def set_S_server(self, S_server):

        self.S_server = S_server

    def get_fsvrg_params(self):

        return (self.n_jks, self.phi_jks, self.nk)

    def _compute_local_fsvrg_params(self):

        self.n_jks = self.get_coord_counts(self.data)
        self.phi_jks = self.n_jks / self.nk

    def get_update(self, global_w, global_grad):

        permutation = np.random.choice(
            self.nk,
            replace=False,
            size=self.nk).tolist()
        local_w = self.get_local(global_w)
        local_grad = self.get_local(global_grad)
        w_i = np.copy(local_w)
        objectives = []

        for i in permutation:
            datum_i = self.model.get_datum(self.data, i)
            local_grad_i = self.get_stochastic_gradient(
                datum_i, local_w)
            grad_i = self.get_stochastic_gradient(
                datum_i, w_i)
            search_direction = self.S_server.get_qn_transform(
                grad_i - local_grad_i) + local_grad
            w_i1 = np.copy(w_i)
            w_i -= self.eta * search_direction

            objectives.append(
                self.get_objective(self.data, w_i))

        new_global_w = np.copy(global_w)

        new_global_w[self.begin:self.end] = np.copy(w_i)

        return (new_global_w, objectives)

    def get_gradient(self, w):

        return self.get_full_gradient(w)
