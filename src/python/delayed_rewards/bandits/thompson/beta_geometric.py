from learner import AbstractLearner
from numpy.random import beta, geometric

class BetaGeometric(AbstractLearner):

    def __init(self, alphas, betas):

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
        self.actions = list(range(self.num_actions))
        self.play_counts = [0] * self.num_actions
        self.rewards = [0] * self.num_actions
        self.waiting = {}
        self.history = []

    def get_action(self):
        
        # Sample and pick highest mean
        geometric_ps = [self._get_geometric_sample(i)
                        for i in self.actions]
        action = max(self.actions, key=lambda i: 1 / geometric_ps[i])

        # Register this action play as waiting for a reward
        self.waiting[len(self.history)] = (0, action)

        # Enter this round into the game history
        self.history.append((action, None, geometric_ps))
        self.play_counts[action] += 1

        return action

    def update_rewards(self, updates):

        # Increment the delay for each unobserved reward
        for time, (delay, action) in self.waiting.items():
            self.waiting[time] = (delay + 1, action)

        for i in updates:

            # If round i is logged as having an unobserved reward
            if i in self.waiting:
                # Retrieve reward and action
                (delay, action) = self.waiting.pop(i)

                # Update rewards for this action
                self.rewards[action] += delay
            else:
                if i > len(self.history):
                    raise ValueError(
                        'Round ' + str(i) + 'has not yet been played.')
                else:
                    raise ValueError(
                        'The reward for round ' + str(i) + \
                        ' has already been observed.')

    def _get_geometric_sample(self, action):

        unobserved = {k : (d, a)
                      for k, (d, a) in self.waiting.items()
                      if a == action}
        post_alpha = self.play_counts[action] - \
            len(unobserved) + \
            self.alphas[action]
        post_beta = len(unobserved) * len(self.history) + \
            self.rewards[action] - \
            sum(unobserved.keys()) + \
            self.play_counts[action] - \
            len(unobserved) + \
            self.betas[action]

        return beta(post_alpha, post_beta)

