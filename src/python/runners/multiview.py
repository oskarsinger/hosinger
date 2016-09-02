class OneModelManyServerRunner:

    def __init__(self, model, servers, max_iter):

        self.model = model
        self.servers = servers
        self.max_iter = max_iter

        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        while not self.converged and self.num_iters < self.max_iter:
            finished = False

            while not finished:
                self.num_rounds += 1

                data = [ds.get_data() for ds in self.servers]

                (y_hat, loss) = model.get_update(data)

                finished = any(
                    [ds.finished() for ds in self.servers])
                
            self.converged = model.is_converged()
            self.num_iters += 1
            
            for ds in self.servers:
                ds.refresh()

    def get_status(self):

        return {
            'model': self.model,
            'servers': self.servers,
            'max_iter': self.max_iter,
            'converged': self.converged,
            'num_iters': self.num_iters,
            'num_rounds': self.num_rounds}
