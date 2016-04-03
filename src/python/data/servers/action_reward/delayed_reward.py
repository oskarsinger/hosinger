from action_reward import AbstractActionRewardServer

class DelayedRewardServer(AbstractActionRewardServer):

    def __init__(self, reward_func, delay_func, incremental=False):
        
        self._reward_func = reward_func
        self._delay_func = delay_func
        self._incremental = incremental
        
        self._data = []
        self._history = []

    def get_rewards(self, action):
        
        reward = self._reward_func(action)
        delay = self._delay_func(action)

        self._data.append((reward,delay))
        self._history.append((action, reward, delay))

        updates = []

        for i, (r, d) in enumerate(self._data):
            
            if d >= -1:
                d = d - 1
                self._data[i] = (r, d)

                if self._incremental:
                    updates.append({'value':r, 'id':i})
                elif d == -1:
                    updates.append({'value':r, 'id':i})

        return updates

    def get_status(self):

        return {
            'data': self._data,
            'history': self._history,
            'reward_func': self._reward_func,
            'delay_func': self._delay_func
        }
