class OneModelOneServerRunner:

    def __init__(self, model, server, max_iter):

        self.model = model
        self.server = server
        self.max_iter = max_iter

        self.converged = False
        self.num_iters = 0
        self.num_rounds = 0

    def run(self):

        while not self.converged and self.num_iters < self.max_iter:
            while not self.server.finished():
                self.num_rounds += 1

                data = self.server.get_data()

                (y_hat, loss) = model.get_update(data)

            self.converged = model.is_converged()
            self.num_iters += 1

            self.server.refresh()

    def get_status(self):

        return {
            'model': self.model,
            'server': self.server,
            'max_iter': self.max_iter,
            'converged': self.converged,
            'num_iters': self.num_iters,
            'num_rounds': self.num_rounds}
