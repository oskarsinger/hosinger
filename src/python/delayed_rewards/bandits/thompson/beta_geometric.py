from learner import AbstractLearner
from numpy.random import beta, geometric

class BetaGeometric(AbstractLearner):

    def __init(self, num_actions, alpha, beta):

        self._num_actions = num_actions
        self._alpha = alpha
        self._beta = beta

        self._actions = list(range(num_actions))
        self._wins_losses = [(0,0)] * num_actions
        self._is_waiting = False
        self._history = []

    def get_status(self):

        return {
            'alpha': self._alpha,
            'beta': self._beta,
            'posteriors': self._wins_losses,
            'waiting': self._is_waiting,
            'actions': self._actions,
            'history': self._history
        }

    def get_action(self):

        print "Stuff"
