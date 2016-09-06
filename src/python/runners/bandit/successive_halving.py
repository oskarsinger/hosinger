class FiniteSuccessiveHalvingRunner:

    def __init__(self,
        arms, 
        servers,
        outer_num_rounds, 
        inner_num_rounds,
        max_size, 
        min_size,
        eta=3):

        self.arms = arms
        self.servers = servers
        self.outer_num_rounds = outer_num_rounds
        self.inner_num_rounds = inner_num_rounds
        self.max_size = max_size
        self.min_size = min_size
        self.eta = eta

        self.num_arms = len(self.arms)
        self.still_pull = [True] * self.num_arms
        self.num_pulls = [0] * self.num_arms
        self.winner = None
        self.winner_loss = None

    def run(self):

        print 'Running SuccessiveHalving for', self.outer_num_rounds, 'rounds'

        for i in xrange(self.outer_num_rounds):

            print 'SuccessiveHalving round', i

            losses = {i : 0
                      for i in xrange(self.num_arms)
                      if self.still_pull[i]}
            n = self.num_arms * self.eta**(-i)
            r = int(self.inner_num_rounds * self.eta**(-i))

            for j in xrange(r):

                # TODO: abstract this to a data selection function
                data = [ds.get_data() for ds in self.servers]

                for k in losses.keys():
                    # TODO: figure out validation loss instead of this crap
                    losses[k] += self.arms[k].update(data)[1]

                    self.num_pulls[k] += 1

            sigma = sorted(
                losses.items(), 
                key=lambda x: x[1]) 
            comparator_j = int(n / self.eta)

            for j, l in sigma[comparator_j:]:
                self.still_pull[j] = False

            self.winner = sigma[0][0]
            self.winner_loss = sigma[0][1]

    def get_status(self):

        return {
            'winner': self.winner,
            'winner_loss': self.winner_loss,
            'arms': self.arms,
            'servers': self.servers,
            'max_size': self.max_size,
            'min_size': self.min_size,
            'eta': self.eta,
            'still_pull': self.still_pull,
            'num_pulls': self.num_pulls, 
            'num_arms': self.num_arms}

class InfiniteSuccessiveHalvingRunner:

    def __init__(self):

        'poop'

    def run(self):

        'poop'

def _get_validation_loss(ds_list, arms, num_points):

    return 'poop'
