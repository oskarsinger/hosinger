from learner import AbstractLearner
from numpy.random import beta, geometric

class BetaGeometric(AbstractLearner):

    def __init(self, num_actions, alpha, beta):

        self.num_actions = num_actions
        self.alpha = alpha
        self.beta = beta

        self.action_counts = [0] * self.num_actions
        self.wins_losses = [(0,0)] * num_actions
        self.waiting = {i : {} for i in self._actions}
        self.history = []

    def get_status(self):

        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'posteriors': self.wins_losses,
            'waiting': self.waiting,
            'actions': self.actions,
            'history': self.history
        }

    def get_action(self):
        
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
