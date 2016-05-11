from gram import AbstractGramServer
from optimization.utils import get_gram

import numpy as np

class BatchGramServer(AbstractGramServer):

    def __init__(self, loader, reg):
        self.loader = loader
        self.reg = reg

    def get_batch_and_gram(self):

        X = self.loader.get_datum()
        gram = get_gram(X, reg=self.reg)

        if not np.isscalar(X):
            gram = gram / X.shape[0]

        return (X, gram)

    def rows(self):

        return self.loader.rows()

    def cols(self):

        return self.loader.cols()

    def get_status(self):

        return "Stuff"
