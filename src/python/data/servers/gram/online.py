import numpy as np

from drrobert.data_structures import FixedLengthQueue as FLQ
from optimization.utils import get_gram as gg

class SumGramServer:

    def __init__(self, reg=0.1):

        self.reg = reg

        self.gram = None
        self.num_rounds = 0
        self.num_examples = 0

    def get_gram(self, batch):

        update = gg(batch, reg=self.reg)

        if self.gram is None:
            self.gram = np.copy(update)
        else:
            self.gram += update

        self.num_examples += batch.shape[0]
        self.num_rounds += 1

        return np.copy(self.gram) / self.num_examples

    def get_status(self):

        return {
            'reg': self.reg,
            'num_rounds': self.num_rounds,
            'num_examples': self.num_examples,
            'gram': self.gram}

class BoxcarGramServer:

    def __init__(self, window=1, reg=0.1):

        self.window = window
        self.reg = reg

        self.q = FLQ(self.window)
        self.gram = None
        self.num_rounds = 0
        self.num_examples = 0

    def get_gram(self, batch):

        update = gg(batch, reg=self.reg)

        if self.gram is None:
            self.gram = np.copy(update)
        elif self.q.is_full():
            self.gram += update
            self.gram -= self.q.get_items()[0]
        else:
            self.gram += update

        self.num_rounds += 1
        self.q.enqueue(update)

        # Compute normalization constant
        num_examples = sum([item.shape[0] 
                            for item in self.q.get_items()])

        return np.copy(self.gram) / num_examples

    def get_status(self):

        return {
            'window': self.window,
            'reg': self.reg,
            'queue': self.q,
            'gram': self.gram,
            'num_rounds': self.num_rounds}

class ExpGramServer:

    def __init__(self, weight=0.9, reg=0.1):

        self.weight = weight
        self.reg = reg

        self.gram = None
        self.num_rounds = 0

    def get_gram(self, batch):

        if self.gram is None:
            cols = batch.shape[1]
            self.gram = np.zeros((cols, cols))

        new_gram = self._get_gram(batch)

        self.gram *= self.weight
        self.gram += new_gram
        self.num_rounds += 1

        return np.copy(self.gram)

    def _get_gram(self, batch):

        n = batch.shape[0]

        return gg(batch, reg=self.reg) / n

    def get_status(self):

        return {
            'weight': self.weight,
            'reg': self.reg,
            'num_rounds': self.num_rounds,
            'gram': self.gram}
