from abc import ABCMeta, abstractmethod

class AbstractExperiment:

    @abstractmethod
    def get_round(self):
        pass

    @abstractmethod
    def get_status(self):
        pass
