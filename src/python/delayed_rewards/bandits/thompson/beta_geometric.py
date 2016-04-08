from learner import AbstractLearner
from numpy.random import beta, geometric

class BetaGeometric(AbstractLearner):

    def __init(self, num_actions, alpha, beta):

        self._num_actions = num_actions
        self._alpha = alpha
        self._beta = beta

        self._actions = list(range(num_actions))
        self._wins_losses = [(0,0)] * num_actions
        self_waiting = {i : [] for i in self._actions}
        self._history = []

    def get_status(self):

        return {
            'alpha': self._alpha,
            'beta': self._beta,
            'posteriors': self._wins_losses,
            'waiting': self._waiting,
            'actions': self._actions,
            'history': self._history
        }

    def get_action(self):
        
        geometric_ps = [self._beta_sample(i)
                        for i in self._actions]
        action = max(self._actions, key=lambda i: bernoulli_ps[i])

        self._history.append((action, None, geometric_ps[action]))

        return action

    def update_rewards(self, updates):

        for (key, value) in updates.items():
            (action, blank, geometric_p) = self._history[key]
            self._history[key] = (action, value, geometric_p)

            (w, l) = self._wins_losses[action]

            if value:
                w += 1
            else:
                l += 1

            self._wins_losses[action] = (w,l)

    def _geometric_sample(self, action):

        (w, l) = self._wins_losses[action]
        post_alpha = self._alpha + w
        post_beta = self._beta + l
