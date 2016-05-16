from gram import AbstractGramServer
from optimization.utils import get_gram as gg

import numpy as np

class BatchGramServer(AbstractGramServer):

    def __init__(self, reg):
        self.reg = reg

    def get_gram(self, batch):

        n = batch.shape[0]

        return gg(batch, reg=self.reg) / n

    def get_status(self):

        return {
            'reg': self.reg}
