import numpy as np

from drrobert.misc import unzip
from optimization.qnservers import StaticDiagonalServer as SDS
from linal.utils import get_safe_power as get_sp

class FSVRG:

    def __init__(self,
        model,
        servers,
        max_rounds=10,
        h=0.01,
        init_params=None):

        self.model = model
        self.servers = servers
        self.num_nodes = len(self.servers)
        self.get_gradient = self.model.get_gradient
        self.get_error = self.model.get_error
        self.get_projection = self.model.get_projection
        self.max_rounds = max_rounds
        self.h = h
        self.init_params = init_params

        self.w = None
        self.nodes = [FSVRGNode(
                        ds,
                        self.get_gradient,
                        self.get_error,
                        i,
                        h=self.h)
                      for (i, ds) in enumerate(self.servers)]
        self.S_servers = None
        self.A_server = None

        self._compute_S_and_A_servers()

        self.errors = []

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
            (ws, errors) = unzip(updates)
            agg = self._get_aggregate_grad(ws, w_t)
            # TODO: Make sure this is a good place to project
            # TODO: Probably need to do a round of communication for projection since it is most likely data-dependent
            # w_t = self.get_projection(w_t + agg)
            w_t = w_t + agg

            self.errors.append(errors)

        self.w = w_t

    def _get_aggregate_grad(self, ws, w_t):

        weighted = sum(
            [(float(nk) / self.n) * (w_k - w_t)
             for (nk, w_k) in zip(self.nks, ws)])

        return self.A_server.get_qn_transform(
            sum(weighted))

    def _compute_S_and_A_servers(self):

        (njks, phi_jks, self.nks) = unzip(
            [n.get_fsvrg_params()
             for n in self.nodes])
        self.n = sum(self.nks)
        njs = sum(njks)
        phi_js = njs / self.n
        sjk_invs = [(phi_jk / phi_js)[:,np.newaxis]
                    for phi_jk in phi_jks]
        self.S_servers = [SDS(sjk_inv)
                          for sjk_inv in sjk_invs]

        for (n, S_s) in zip(self.nodes, self.S_servers):
            n.set_S_server(S_s)

        omega_js = np.vstack(
            (njk[:,np.newaxis] != 0).astype(float) 
            for njk in njks)
        aj_invs = (omega_js / self.num_nodes)

        self.A_server = SDS(aj_invs)

class FSVRGNode:

    def __init__(self,
        server,
        get_gradient,
        get_error,
        id_number,
        h=0.1):

        self.server = server
        self.get_error = get_error
        self.id_number = id_number

        self.nk = self.server.rows()
        self.p = self.server.cols()
        # TODO: replace begin/end with arbitrary index vector
        self.begin = self.id_number * self.p
        self.end = self.begin + self.p
        self.get_local = lambda x: x[self.begin:self.end]
        self.data = self.server.get_data()
        (self.A, self.b) = self.data
        (self.n_jks, self.phi_jks) = [None] * 2

        self._compute_local_fsvrg_params()

        self.get_full_gradient = lambda w: get_gradient(
            self.data, w)
        self.get_stochastic_gradient = get_gradient
        self.eta = h / self.nk

    def set_S_server(self, S_server):

        self.S_server = S_server

    def get_fsvrg_params(self):

        return (self.n_jks, self.phi_jks, self.nk)

    def _compute_local_fsvrg_params(self):

        self.n_jks = np.sum(
            (self.data[0] != 0).astype(float),
            axis=0)
        self.phi_jks = self.n_jks / self.nk

    def get_update(self, global_w, global_grad):
        
        permutation = np.random.choice(
            self.nk,
            replace=False,
            size=self.nk).tolist()
        local_w = self.get_local(global_w)
        local_grad = self.get_local(global_grad)
        w_i = np.copy(local_w)

        for i in permutation:
            datum_i = (self.A[i,:][np.newaxis,:], self.b[i])
            local_grad_i = self.get_stochastic_gradient(
                datum_i, local_w)
            grad_i = self.get_stochastic_gradient(
                datum_i, w_i)
            print np.linalg.norm(grad_i - local_grad_i)
            search_direction = self.S_server.get_qn_transform(
                grad_i - local_grad_i) + local_grad
            w_i -= self.eta * search_direction

        error = self.get_error(self.data, w_i)
        new_global_w = np.copy(global_w)

        new_global_w[self.begin:self.end] = np.copy(w_i)


        return (new_global_w, error)

    def get_gradient(self, w):

        return self.get_full_gradient(w)
