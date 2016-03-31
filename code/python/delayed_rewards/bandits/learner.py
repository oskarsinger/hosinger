from abc import ABCMeta, abstractmethod

class AbstractLearner:
    __metaclass__=ABCMeta

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update_rewards(self, value):
        pass

    @abstractmethod
    def get_status(self):
        pass
