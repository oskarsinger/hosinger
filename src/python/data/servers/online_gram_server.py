import numpy as np

from gram_server import AbstractGramServer
from linal.utils import get_lms

class ExpOnlineGramServer(AbstractGramServer):

    def __init__(self, data_loader, weight):

        self.dl = data_loader
        self.weight = weight

        cols = self.dl.cols()

        self.gram = np.zeros((cols, cols))
        self.num_rounds = 0

    def get_batch_and_gram(self):

        self.num_rounds += 1

        row = self.dl.get_datum()
        w = (self.weight)**(self.num_rounds)

        self.gram += w * np.dot(row, row.T)

        return (row, np.copy(self.gram))

    def rows(self):

        return self.num_rounds

    def cols(self):

        return self.dl.cols()

    def get_status(self):
        
        return {
            'gram': self.gram,
            'weight': self.weight,
            'num_rounds': self.num_rounds,
            'data_loader': self.dl}

class BoxcarOnlineGramServer(AbstractGramServer):

    def __init__(self, data_loader, length):

        self.dl = data_loader
        self.length = length

        cols = self.dl.cols()
        
        self.grams = []
        self.num_rounds = 0

    def get_batch_and_gram(self):

        self.num_rounds += 1

        row = self.dl.get_datum()
        gram = np.dot(row, row.T)

        if len(self.grams) >= self.length:
            self.grams = self.grams[1:] + [gram]

        return (row, sum(self.grams))

    def rows(self):

        return self.num_rounds

    def cols(self):

        return self.dl.cols()

    def get_status(self):

        return {
            'gram': self.gram,
            'length': self.length,
            'num_rounds': self.num_rounds,
            'data_loader': self.dl}
