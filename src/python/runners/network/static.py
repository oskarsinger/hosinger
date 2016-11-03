import numpy as np

class StaticStructureRunner:

    def __init__(self,
        servers,
        adj_lists,
        learners):

        self.servers = servers
        self.adj_lists = adj_lists
        self.learners = learners

    def run(self):

        adj_stuff = zip(self.servers, self.adj_lists)

        for (s, al) in adj_stuff:
            neighbors = [s for s in self.servers
                         if s.id_num in al]
            s.set_neighbors(neighbors)

        for i in xrange(self.num_rounds):
            actions = [l.get_action() for l in self.learners]
            feedback = [s.get_data(a) for s in self.servers]

            for (l, f) in zip(self.learners, feedback):
                l.set_feedback(f)
