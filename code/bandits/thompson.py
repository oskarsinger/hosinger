import AbstractLearner
from numpy.random import beta

class BetaBernoulli(AbstractLearner):

    def __init__(self, num_actions, alpha, beta):

        self._num_actions = num_actions
        self._alpha = alpha
        self._beta = beta

        self._actions = list(range(actions))
        self._wins_losses = [(0,0)] * num_actions
        self._is_waiting = False
        self._history = []

    def get_status(self):

        return {
            'alpha': self._alpha,
            'beta': self._beta,
            'posteriors': self._wins_losses,
            'waiting': self._is_waiting,
            'history': self._history
        }

    def get_action(self):
        
        if self._is_waiting:
            raise Exception('This learner is still waiting for a reward.')

        self._is_waiting = True

        bernoulli_ps = [self._beta_sample(i)
                        for i in self._actions]
        action = max(self._actions, key=lambda i: bernoulli_ps(i))

        self._history.append((action, None, bernoulli_ps[action]))

        return action

    def update_reward(self, value):

        if not self._is_waiting:
            raise Exception('This learner has not taken an action after receiving' +
            'its most recent reward.')

        self._is_waiting = False

        (action, blank, bernoulli_p) = self._history[-1][0]
        self._history[-1] = (action, value, bernoulli_p)

        (w, l) = self._wins_losses[action]

        if value:
            w = w+1
        else:
            l = l+1

        self._wins_losses[action] = (w,l)

    def _beta_sample(action):
        
        (w, l) = self._wins_losses[action]
        alpha = self._alpha + w
        beta = self._beta + l

        return beta(alpha, beta)
