from numpy.random import choice
from learner import AbstractLearner

import math

class Exp3(AbstractLearner):

    def __init__(self, num_actions, gamma):

        self._num_actions = num_actions
        self._gamma = gamma

        self._weights = [1.0] * num_actions
        self._actions = list(range(num_actions))
        self._is_waiting = False
        self._history = []
        self._num_rounds = 0

    def get_status(self):

        return {
            'gamma': self._gamma,
            'weights': self._weights,
            'waiting': self._is_waiting,
            'history': self._history
        }

    def get_action(self):

        if self._is_waiting:
            raise Exception('This learner is still waiting for a reward.')

        self._is_waiting = True
    
        self._update_weights()

        action = choice(self._actions, p=self._weights)

        self._history.append((action, None, None))

        self._num_rounds = self._num_rounds + 1

        return action

    def update_reward(self, value):

        if not self._is_waiting:
            raise Exception('This learner has not taken an action after receiving' +
            'its most recent reward.')

        self._is_waiting = False

        action = self._history[-1][0]
        self._history[-1] = (action, value, self._weights[action])
        estimated_reward = value / self._weights[action]
        self._weights[action] = self._weights[action] * math.exp(
            estimated_reward * self._gamma / len(self._actions))

    def _update_weights(self):

        weight_sum = sum(self._weights)
        update = lambda w: (1.0 - self._gamma) * (w / weight_sum) + \
                           self._gamma / len(self._weights)

        self._weights = [update(w)
                         for w in self._weights]
