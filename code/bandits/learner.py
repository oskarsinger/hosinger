from abc import ABCMeta, abstractmethod

class AbstractLearner(metaclass=ABCMeta):

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update_reward(self, value):
        pass
