import numpy as np

from data.servers.gram import AbstractOnlineGramServer
from optimization.utils import get_t_regged_gram as get_trg
from global_utils.data_structures import FixedLengthQueue as FLQ

class BoxcarOnlineGramServer(AbstractOnlineGramServer):

    def __init__(self, data_loader, batch_size, reg=0.1):

        super(BoxcarOnlineGramServer, self).__init__(
            data_loader, batch_size, reg)

    def _get_gram(self):

        minibatch = np.array(self.minibatch.get_items())

        return get_trg(minibatch, self.reg)

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
        gram = get_trg(minibatch, self.reg)

        self.gram += w * gram

        return np.copy(self.gram)

    def get_status(self):

        init = super(ExpOnlineGramServer, self).get_status()
        
        return init + {
            'weight': self.weight,
            'gram': self.gram}

