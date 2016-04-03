from abc import ABCMeta, abstractmethod

class AbstractActionRewardServer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_rewards(self, action):
        pass

    @abstractmethod
    def get_status(self):
        pass
