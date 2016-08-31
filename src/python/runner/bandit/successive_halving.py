from math import floor

class FiniteSuccessiveHalvingRunner:

    def __init__(self,
        arms, 
        servers,
        budget, 
        max_size, 
        min_size,
        eta=3):

        self.arms = arms
        self.servers = servers
        self.budget = budget
        self.max_size = max_size
        self.min_size = min_size
        self.eta = eta

        self.num_arms = len(self.arms)

        threshold = float(self.budget) / \
            (self.num_arms * self.max_size)
        i = 0

        while (i+1) * self.eta**(-i) <= threshold:
            i += 1

        self.max_rounds = i
        
        self.still_pull = [True] * self.num_arms
        self.num_pulls = [0] * self.num_arms
        self.num_rounds = 0
        self.winner = None

    def run(self):

        for k in xrange(self.max_rounds):

            losses = {i : 0
                      for i in xrange(self.num_arms)
                      if self.still_pull[i]}
            r = int(floor(self.max_size * 
                self.eta**(k - self.max_rounds)))

            for i in xrange(r):

                data = [ds.get_data() for ds in self.servers]

                for j in losses.keys():
                    # TODO: should I be accumulating or just taking last one?
                    losses[j] += self.arm[j].update(data)[1]

            sigma = sorted(
                losses.items(), 
                key=lambda x: x[1]) 
            comparator_j = int(floor(
                self.num_arms * self.eta**(-k+1)))

            for j, l in sigma[comparator_j:]:
                self.still_pull[j] = False

            self.winner = sigma[0][0]

    def get_status(self):

        return {
            }
