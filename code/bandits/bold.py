class Bold():

    def __init__(self, get_base):

        self._get_base = get_base

        self._no_feedback = {}
        self._bases = [(self._get_base(), True)]

    def process(self, item, rewards):

        """
        The order in which _get_action and _update_subroutines are called
        is important because they both modify the same state-maintaining class 
        variables.
        """

        action = self._get_action(item)

        self._update_bases(rewards)

        return action

    def _get_action(self, item):

        av_subs = [(i, base)
                   for i, (base, available) in enumerate(self._bases)
                   if available]
        sub = None

        if len(av_subs) > 0:
            i, sub = av_subs[0] #maybe switch this to randomly select
        else:
            sub = self._get_base()
            i = len(self._bases)

            self._bases.append((sub, True))

        action = sub.get_action()
        self._bases[i] = (sub, False)

        self._no_feedback[item['id']] = (item, i)

        return action

    def _update_bases(rewards):

        for reward in rewards:
            i = self._no_feedback[reward['id']][1]
            base = self._bases[i]

            base.update(reward['value'])
            self._bases[i] = (base, True)
