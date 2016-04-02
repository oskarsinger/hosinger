import numpy as np

from gram_server import AbstractGramServer
from linal.utils import get_lms

class OnlineGramServer(AbstractGramServer):

    def __init__(self, data_loader, batch_size=1, weight_server=None):

        self.dl = data_loader
        self.batch_size = batch_size
        self.ws = weight_server
        self.weighted = weight_server is not None

        cols = self.dl.cols()
        self.grams = [np.zeros((cols, cols))]
        self.batch = np.zeros((batch_size,cols))
        self.num_rounds = 0

    def get_batch_and_gram(self):

        self.num_rounds += 1

        for i in range(self.batch_size):
            row = self.dl.get_datum()

    def rows(self):

        return self.num_rounds

    def cols(self):

        return self.dl.cols()

    def get_status(self):
        
        print "Some stuff"
