from abc import ABCMeta, abstractmethod

class AbstractCOMID:

    @abstractmethod
    def get_comid_update(self, parameters, gradient):
        pass

    @abstractmethod
    def get_status(self):
        pass
