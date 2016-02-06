class AbstractLearner:

    def get_action(self):
        raise NotImplementedError( 'Implementation of this method is required.')

    def update_reward(self, value):
        raise NotImplementedError( 'Implementation of this method is required.')
