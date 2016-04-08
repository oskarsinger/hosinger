from learner import AbstractLearner
from numpy.random import beta

import bandit_errors as be

class BetaBernoulli(AbstractLearner):

    def __init__(self, num_actions, alpha, beta):

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
        
        if self._is_waiting:
            be.raise_no_reward_error()

        self._is_waiting = True

        bernoulli_ps = [self._beta_sample(i)
                        for i in self._actions]
        action = max(self._actions, key=lambda i: bernoulli_ps[i])

        self._history.append((action, None, bernoulli_ps[action]))

        return action

    def update_rewards(self, value):

        if not self._is_waiting:
            be.raise_no_action_error()

        self._is_waiting = False

        (action, blank, bernoulli_p) = self._history[-1]
        self._history[-1] = (action, value, bernoulli_p)

        (w, l) = self._wins_losses[action]

        if value:
            w += 1
        else:
            l += 1

        self._wins_losses[action] = (w,l)

    def _beta_sample(self, action):
        
        (w, l) = self._wins_losses[action]
        post_alpha = self._alpha + w
        post_beta = self._beta + l

        return beta(post_alpha, post_beta)
