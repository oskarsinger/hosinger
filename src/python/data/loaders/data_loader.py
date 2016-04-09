from abc import ABCMeta, abstractmethod

class AbstractDataLoader:

    @abstractmethod
    def get_datum(self):
        pass

    @abstractmethod
    def get_status(self):
        pass

    @abstractmethod
    def cols(self):
        pass
