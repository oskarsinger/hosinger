from abc import ABCMeta, abstractmethod

class AbstractDataServer:
    __metaclass__=ABCMeta

    @abstractmethod
    def get_rewards(self, action):
        pass

    @abstractmethod
    def get_data_point(self):
        pass

    @abstractmethod
    def get_status(self):
        pass
