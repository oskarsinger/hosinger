from abc import ABCMeta, abstractmethod

class AbstractSGD:

    @abstractmethod
    def get_gradient(self, **kwargs):
        pass

    @abstractmethod
    def get_projection(self, **kwargs):
        pass
