from learner import AbstractLearner
from numpy.random import beta

import bandit_errors as be

class BetaBernoulli(AbstractLearner):

    def __init__(self, alphas, betas):

        if not len(alphas) == len(betas):
            raise ValueError(
                'Parameters alphas and betas must have same length.')

        if not all([a > 0 for a in alphas]):
            raise ValueError(
                'All elements of parameter alphas must be > 0.')

        if not all([b > 0 for b in betas]):
            raise ValueError(
                'All elements of parameter betas must be > 0.')

        self.alphas = alphas
        self.betas = betas

        self.num_actions = len(alphas)
        self.actions = list(range(num_actions))
        self.wins_losses = [(0,0)] * num_actions
        self.is_waiting = False
        self.history = []

    def get_status(self):

        return {
            'num_actions': self.num_actions,
            'alphas': self.alphas,
            'beta': self.beta,
            'wins_losses': self.wins_losses,
            'waiting': self.is_waiting,
            'actions': self.actions,
            'history': self.history
        }

    def get_action(self):
        
        if self.is_waiting:
            be.raise_no_reward_error()

        self.is_waiting = True

        bernoulli_ps = [self._get_beta_sample(i)
                        for i in self.actions]
        action = max(self.actions, key=lambda i: bernoulli_ps[i])

        self.history.append((action, None, bernoulli_ps))

        return action

    def update_rewards(self, value):

        if not self.is_waiting:
            be.raise_no_action_error()

        self.is_waiting = False

        (action, blank, bernoulli_ps) = self.history[-1]
        self.history[-1] = (action, value, bernoulli_ps)

        (w, l) = self.wins_losses[action]

        if value:
            w += 1
        else:
            l += 1

        self.wins_losses[action] = (w,l)

    def _get_beta_sample(self, action):
        
        (w, l) = self.wins_losses[action]
        post_alpha = self.alphas[action] + w
        post_beta = self.betas[action] + l

        return beta(post_alpha, post_beta)
