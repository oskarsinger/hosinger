class Bold():

    def __init__(self, get_learner):

        self._get_learner = get_learner

        self._no_feedback = {}
        self._learners = [(self._get_learner(), True)]

    def process(self, item, rewards):

        """
        The order in which _get_action and _update_subroutines are called
        is important because they both modify the same state-maintaining class 
        variables.
        """

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
            sub = self._get_learner()
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
