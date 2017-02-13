import numpy as np

from optimization.optimizers import GradientOptimizer as GO
from clustering import BallTree as BT

class ZeroOrderSwarm:

    def __init__(self,
        num_units,
        server,
        optimizers=None,
        get_neighbors=None):

        self.num_units = num_units
        self.server = server
        
        if optimizers is None:
            optimizers = [GO() for i in xrange(self.num_units)]

        self.optimizers = optimizers

        if get_neighbors is None:
            get_neighbors = lambda X: BallTree(X).fit().get_neighbors()

        self.get_neighbors = get_neighbors

    def run(self):

        print 'Poop'
