class Bold():

    def __init__(self, get_learner, num_actions):

        self._get_learner = get_learner
        self._num_actions = num_actions

        self._no_feedback = {}
        self._learners = [(self._get_learner(), True)]

    def process(self, item, rewards):

        action = self._get_action(item)

        self._update_learners(rewards)

        return action

    def _get_action(self, item):

        av_subs = [(i, learner)
                   for i, (learner, available) in enumerate(self._learners)
                   if available]
        sub = None

        if len(av_subs) > 0:
            i, sub = av_subs[0] #maybe switch this to randomly select
        else:
            sub = self._get_learner(self._num_actions)
            i = len(self._learners)

            self._learners.append((sub, True))

        action = sub.get_action()
        self._learners[i] = (sub, False)

        self._no_feedback[item['id']] = (item, i)

        return action

    def _update_learners(rewards):

        for reward in rewards:
            i = self._no_feedback[reward['id']][1]
            learner = self._learners[i]

            learner.update_reward(reward['value'])
            self._learners[i] = (learner, True)
