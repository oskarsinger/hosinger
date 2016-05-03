from abc import ABCMeta, abstractmethod

from global_utils.data_structures import FixedLengthQueue as FLQ

class AbstractGramServer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_batch_and_gram(self):
        pass

    @abstractmethod
    def rows(self):
        pass

    @abstractmethod
    def cols(self):
        pass

    @abstractmethod
    def get_status(self):
        pass

class AbstractOnlineGramServer(AbstractGramServer):

    def __init__(self, data_loader, batch_size, reg):

        self.dl = data_loader
        self.batch_size = batch_size
        self.reg = reg

        self.minibatch = FLQ(self.batch_size)
        self.no_more_data = False
        self.num_rounds = 0

    def get_batch_and_gram(self):

        self.num_rounds += 1 

        self._update_minibatch()
        
        gram = self._get_gram()
        minibatch = self.minibatch.get_items()

        output = None if self.no_more_data else (minibatch, gram)

        return output

    def _update_minibatch(self):

        candidates = [self.dl.get_datum()
                      for i in xrange(self.batch_size)]
        non_null = [datum for datum in candidates
                    if datum is not None]

        for datum in non_null:
            self.minibatch.enqueue(datum)

        self.no_more_data = len(non_null) > 0

    def rows(self):

        return self.num_rounds

    def cols(self):

        return self.dl.cols()

    def get_status(self):

        return {
            'data_loader': self.dl,
            'batch_size': self.batch_size,
            'reg': self.reg,
            'minibatch': self.minibatch,
            'no_more_data': self.no_more_data,
            'num_rounds': self.num_rounds}

    @abstractmethod
    def _get_gram(self):
        pass
