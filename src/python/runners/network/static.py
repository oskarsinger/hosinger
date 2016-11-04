import numpy as np
from drrobert.network import get_erdos_renyi as get_er

class SyntheticStaticStructureRunner:

    def __init__(self):

        num_nodes = 50
        self.servers = ['Poop' for i in xrange(num_nodes)]
        # TODO: consider converting input adj_matrix to adj lists
        self.adj_matrix = get_er(nodes, 0.05)
        self.learners = ['Poop' for i in xrange(num_nodes)]

    # TODO: change this to reflect adj_matrix instead of lists
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
