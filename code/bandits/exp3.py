
class Exp3(AbstractLearner):

    def __init__(self, num_actions, gamma):

        self._num_actions = num_actions
        self._gamma = gamma

        self._weights = [1.0] * num_actions
        self._is_waiting = False

    def get_action(self):

        print 'Do something'

    def update_reward(self, value):

        print 'Do something'
