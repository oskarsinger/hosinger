from learner import AbstractLearner
from bandit_errors import *
from itertools import izip

import math

class UCB1(AbstractLearner):

    def __init__(self, num_actions):

        self._num_actions = num_actions

        self._reward_sums = [0] * num_actions
        self._num_plays = [0] * num_actions
        self._ucbs = [0] * num_actions
        self._history = []
        self._num_rounds = 0
        self._is_waiting = False

    def get_status(self):

        return {
            'ucbs': self._ucbs,
            'actions': list(range(self._num_actions)),
            'waiting': self._is_waiting,
            'history': self._history
        }

    def get_action(self):

        if self._is_waiting:
            raise_no_reward_error()

        self._is_waiting = True

        action = None

        if self._num_rounds < self._num_actions:
            action = self._num_rounds #maybe randomize this?
        else:
            args = izip(self._reward_sums, self._num_plays)
            self._ucbs = [self._get_ucb(reward_sum, num_plays)
                          for reward_sum, num_plays in args]
            action = max(range(self._num_actions), key=lambda i: self._ucbs[i])

        self._num_rounds = self._num_rounds + 1
        self._num_plays[action] = self._num_plays[action] + 1
        self._history.append((action, None, self._ucbs[action]))

        return action

    def update_rewards(self, value):

        if not self._is_waiting:
            raise_no_action_error()

        self._is_waiting = False

        (action, blank, ucb) = self._history[-1]
        self._history[-1] = (action, value, ucb)
        self._reward_sums[action] = self._reward_sums[action] + value

    def _get_ucb(self, reward_sum, num_plays):

        upper_bound = math.sqrt(2 * math.log(self._num_rounds + 1) / num_plays)

        return reward_sum / num_plays + upper_bound
