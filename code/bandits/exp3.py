from numpy.random import choice

import math

class Exp3(AbstractLearner):

    def __init__(self, num_actions, gamma):

        self._gamma = gamma

        self._weights = [1.0] * num_actions
        self._actions = list(range(num_actions))
        self._num_rounds = 0
        self._is_waiting = False
        self._play_log = []
        self._cumulative_reward = 0

    def get_action(self):

        if self._is_waiting:
            raise Exception('This learner is still waiting for a reward.')
    
        self._update_weights()
        action = choice(self._actions, p=self._weights)
        self._play_log.append((action, None, None))
        self._num_rounds = self._num_rounds + 1
        self._is_waiting = True

        return action

    def update_reward(self, value):

        if not self._is_waiting:
            raise Exception('This learner has not taken an action after receiving' +
            'its most recent reward.')

        action = self._play_log[-1][0]

        self._is_waiting = False
        self._play_log[-1] = (action, value, self._weights[action])

        estimated_reward = value / self._weights[choice]
        self._weights[action] *= math.exp(
            estimated_reward * self._gamma / len(self._actions))

    def _update_weights(self):

        weight_sum = sum(self._weights)
        update = lambda w: (1.0 - self._gamma) * (w / weight_sum) + \
                           self._gamma / len(weights)

        self._weights = [update(w)
                         for w in weights]
