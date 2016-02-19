from learner import AbstractLearner

class BOLD(AbstractLearner):

    def __init__(self, get_learner, num_actions):

        self._get_learner = get_learner
        self._num_actions = num_actions

        self._learners = [(self._get_learner(num_actions), True)]
        self._history = []
        self._num_rounds = 0

    def get_status(self):

        return {
            'learners': self._learners,
            'history': self._history
        }

    def get_action(self):

        action = self._get_action(self._num_rounds)

        self._num_rounds = self._num_rounds + 1

        return action

    def _get_action(self, item):

        availables = [(i, learner)
                      for i, (learner, available) in enumerate(self._learners)
                      if available]
        chosen_learner = None
        learner_id = None

        if len(availables) > 0:
            learner_id, chosen_learner = availables[0]
        else:
            chosen_learner = self._get_learner(self._num_actions)
            learner_id = len(self._learners)

            self._learners.append((chosen_learner, True))

        action = chosen_learner.get_action()
        self._learners[learner_id] = (chosen_learner, False)

        self._history.append((action, learner_id, None))

        return action

    def update_reward(self, rewards):

        for reward in rewards:
            (action, learner_id, blank) = self._history[reward['id']]
            learner = self._learners[learner_id][0]

            learner.update_reward(reward['value'])

            self._history[reward['id']] = (action, learner_id, reward['value'])
            self._learners[learner_id] = (learner, True)
