from action_reward import AbstractActionRewardServer

class StochasticDelayOneTimeRewardServer(AbstractActionRewardServer):

    def __init__(self, reward_func, delay_func):
        
        self.reward_func = reward_func
        self.delay_func = delay_func
        
        self.waiting = {}
        self.history = []

    def get_rewards(self, action):
        
        # Generate reward and delay
        reward = self.reward_func(action)
        delay = self.delay_func(action)

        # Add reward and delay to log of waiting items
        # Key is the zero-based timestep of generation
        self.waiting[len(self.history)] = ((reward,delay))

        # Add reward, delay, and action to history
        self.history.append((action, reward, delay))

        updates = []

        # Iterate through timesteps with unresolved reward
        for t in self.waiting.keys():

            (r, d) = self.waiting[t]
            
            # If generation timestep plus delay is current
            if t + d == len(self.history) - 1:
                # Include this reward and its timestep in updates
                updates.append({'value': r, 'id': t})

                # Remove this timestep from waiting log
                del self.waiting[t]

        return updates

    def get_status(self):

        return {
            'waiting': self.waiting,
            'history': self.history,
            'reward_func': self.reward_func,
            'delay_func': self.delay_func}

class StochasticDelayAsRewardServer(AbstractActionRewardServer):

    def __init__(self, delay_func):
        self.delay_func = delay_func

        self.waiting = {}
        self.history = []

    def get_rewards(self, action):

        delay = self.delay_func(action)

        # Add delay to log of waiting items
        # Key is the zero-based timestep of generation
        self.waiting[len(self.history)] = (delay)

        # Add delay action to history
        self.history.append((delay, action))

        updates = []

        for t in self.waiting.keys():

            d = self.waiting[t]
            
            # If generation timestep plus delay is current
            if t + d == len(self.history) - 1:
                # Include this timestep in updates
                updates.append(t)

                # Remove this timestep from waiting log
                del self.waiting[t]

        return updates

    def get_status(self):

        return {
            'waiting': self.waiting,
            'history': self.history,
            'delay_func': self.delay_func}

