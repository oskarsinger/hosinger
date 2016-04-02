from abc import ABCMeta, abstractmethod

class AbstractGramServer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_batch_and_gram(self):
        pass

    @abstractmethod
    def rows():
        pass

    @abstractmethod
    def cols():
        pass

    @abstractmethod
    def get_status(self):
        pass
