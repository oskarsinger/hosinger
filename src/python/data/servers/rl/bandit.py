import numpy as np

class BanditServer:

    def __init__(self, loader):

        self.loader = loader

        self.num_rounds = 0

    def set_action(self, action):

        self.loader.set_action(action)

    def get_reward(self):

        return self.loader.get_reward()

    def cols(self):

        return self.loader.cols()

    def rows(self):

        return self.loader.rows()

    def name(self):

        return 'BanditServer'
