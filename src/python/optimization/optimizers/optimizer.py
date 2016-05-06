from abc import ABCMeta, abstractmethod

class AbstractOptimizer:
    __metaclass__=ABCMeta

    @abstractmethod
    def get_update(self, parameters, gradient, eta):
        pass

    @abstractmethod
    def get_status(self):
        pass
