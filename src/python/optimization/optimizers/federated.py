import numpy as np

class FederatedOptimizer:

    def __init__(self):

        self.num_rounds = 0

        self.A = None
        self.S = None

    def get_update(self, parameters, gradient, eta):

        self.num_rounds += 1

        print 'Poop'
