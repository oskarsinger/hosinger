import numpy as np

# TODO: consider using scipy sparse LA stuff
# REMEMBER: A and S servers should be fed inverse of A and S
class FSVRG:

    def __init__(self,
        servers,
        get_gradient,
        get_error,
        get_projection,
        A_server,
        S_servers,
        num_rounds=10,
        h=0.01,
        init_params=None):

        self.servers = servers
        self.num_nodes = len(self.servers)
        self.get_gradient = get_gradient
        self.get_error = get_error
        self.get_projection = get_projection
        self.A_server = A_server
        self.S_servers = S_servers
        self.max_rounds = max_rounds
        self.eta = eta
        self.init_params = init_params

        self.n_ks = [ds.rows()
                     for ds in self.servers]
        self.n = sum(self.n_ks)
        self.w = None

        node_stuff = zip(
            servers,
            self.S_servers)

        self.nodes = [FSVRGNode(
                        ds,
                        S_s,
                        self.gradient,
                        self.get_error,
                        h=self.h)
                      for (ds, S_s) in node_stuff]
        self.errors = []

    def get_parameters(self):

        if self.w is None:
            raise Exception(
                'Parameters have not been computed.')
        
        return np.copy(self.w)

    def compute_parameters(self, parameters, gradient, eta):

        w_t = np.copy(self.init_params)

        for i in xrange(self.max_rounds):
            local_grads = [n.get_gradient(np.copy(w_t))
                           for n in self.nodes]
            grad_t = sum(local_grads) / self.num_nodes
            updates = [n.get_update(
                            np.copy(w_t),
                            np.copy(grad_t))
                       for n in self.nodes]
            (ws, errors) = unzip(updates)
            agg = self._get_aggregate_grad(ws)
            w_t = w_t + agg

        self.w = w_t

    def _get_aggregate_grad(self, ws):

        weighted = [(float(n_k) / self.n) * w_k
                    for (n_k, w_k) in zip(self.n_ks, ws)]

        return self.A_server.get_qn_transform(
            sum(weighted))

class FSVRGNode:

    def __init__(self,
        server,
        S_server,
        get_gradient,
        get_error,
        h=0.1):

        self.server = server
        self.get_error = get_error

        self.n = self.server.rows()
        self.data = self.server.get_data()
        self.get_full_gradient = lambda w: get_gradient(
            self.data, w)
        self.get_stochastic_gradient = get_gradient
        self.eta = h / self.n

    def get_update(self, global_w, global_grad):
        
        permutation = np.random.choice(
            self.n,
            replace=False,
            self.n).tolist()
        local_w = np.copy(global_w)

        for i in permutation:
            datum_i = self.data[i,:]
            global_grad_i = self.get_stochastic_gradient(
                datum_i, global_w)
            local_grad_i = self.get_stochastic_gradient(
                datum_i, local_w)
            search_direction = self.S_server.get_qn_transform(
                local_grad_i - global_grad_i) + global_grad
            local_w -= self.eta * search_direction

        return local_w

    def get_gradient(self, w):

        return self.get_full_gradient(w)
