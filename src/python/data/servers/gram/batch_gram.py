from gram import AbstractGramServer

from optimization.utils import get_t_regged_gram

class BatchGramServer(AbstractGramServer):

    def __init__(self, loader, reg):
        self.loader = loader

    def get_batch_and_gram(self):

        X = loader.get_datum()
        gram = get_t_regged_gram(X.T, X, reg)

        return (X, gram)

    def rows(self):

        return self.loader.rows()

    def cols(self):

        return self.loader.cols()

    def get_status(self):

        return "Stuff"
