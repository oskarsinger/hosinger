import numpy as np

from data.servers.gram import AbstractOnlineGramServer
from optimization.utils import get_gram
from global_utils.data_structures import FixedLengthQueue as FLQ

class BoxcarOnlineGramServer(AbstractOnlineGramServer):

    def __init__(self, data_loader, batch_size, reg=0.1):

        super(BoxcarOnlineGramServer, self).__init__(
            data_loader, batch_size, reg)

    def _get_gram(self):

        batch = np.array(self.minibatch.get_items())

        return get_gram(batch, reg=self.reg) / batch_size

class ExpOnlineGramServer(AbstractOnlineGramServer):

    def __init__(self, data_loader, batch_size, weight, reg=0.1):

        super(ExpOnlineGramServer, self).__init__(
            data_loader, batch_size, reg)

        self.weight = weight

        cols = self.dl.cols()

        self.gram = np.zeros((cols, cols))

    def _get_gram(self):

        w = (self.weight)**(self.num_rounds)

        minibatch = np.array(self.minibatch.get_items())
        new_gram = get_gram(minibatch, reg=self.reg)

        self.gram += w * new_gram

        return np.copy(self.gram)

    def get_status(self):

        init = super(ExpOnlineGramServer, self).get_status()
        
        return init + {
            'weight': self.weight,
            'gram': self.gram}

