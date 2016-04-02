from gram_server import AbstractGramServer

from optimization.utils import get_t_regged_gram

class BatchGramServer(AbstractGramServer):

    def __init__(self, X, reg):
        self.X = X
        self.X_gram = get_t_regged_gram(X.T, X, reg)
        (self.n, self.p) = X.shape

    def get_batch_and_gram(self):

        return (self.X, self.X_gram)

    def rows():

        return self.n

    def cols():

        return self.p

    def get_status(self):

        return "Stuff"
