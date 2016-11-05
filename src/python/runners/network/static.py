import numpy as np
import drrobert.network as dn

from data.loaders.synthetic import VertexWithExposureLoader as VWEL
from learners.network import NetworkInterferenceLearner as NIFL

class SyntheticStaticStructureRunner:

    def __init__(self):

        num_nodes = 20
        self.burn_in = 100
        self.servers = [VWEL() 
                        for i in xrange(num_nodes)]
        self.adj_matrix = dn.get_erdos_renyi(nodes, 0.05, sym=True)
        self.adj_lists = dn.get_adj_lists(self.adj_matrix)
        self.learners = [NIFL(i, self.adj_matrix, self.burn_in)
                         for i in xrange(num_nodes)]

    def run(self):

        adj_stuff = zip(self.servers, self.adj_lists)

        for (s, al) in adj_stuff:
            neighbors = [s for s in self.servers
                         if s.id_num in al]
            s.set_neighbors(neighbors)

        for i in xrange(self.num_rounds):
            actions = [l.get_action() for l in self.learners]

            for (s, a) in zip(self.servers, actions):
                s.set_action(a)

            feedback = [s.get_data() for s in self.servers]

            for (l, f) in zip(self.learners, feedback):
                l.set_feedback(f)
