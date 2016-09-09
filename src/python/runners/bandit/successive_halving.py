import drrobert.debug as drdb

from multiprocessing import Pool
from drrobert.misc import unzip

class FiniteSuccessiveHalvingRunner:

    def __init__(self,
        arms, 
        server,
        outer_num_rounds, 
        inner_num_rounds,
        max_size, 
        min_size,
        num_processes=12,
        eta=3):

        self.arms = arms
        self.server = server
        self.outer_num_rounds = outer_num_rounds
        self.inner_num_rounds = inner_num_rounds
        self.max_size = max_size
        self.min_size = min_size
        self.num_processes = num_processes
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
            num_arms_i = self.num_arms * self.eta**(-i)
            r_i = int(self.inner_num_rounds * self.eta**(-i))
            num_processes_i = min([12, num_arms_i])

            print '\tRunning inner loop for', r_i, 'rounds'
            for j in xrange(r_i):

                data = self.server.get_data()

                print 'Inside SH with data length', len(data)

                # Parallelize this loop
                p = Pool(num_processes_i)
                keys = losses.keys()
                num_keys = len(keys)
                results = {}
                k = 0
                
                while k * num_processes_i < num_keys:
                    begin = k * num_processes_i
                    end = begin + num_processes_i
                    current = {l : None
                               for l in keys[begin:end]}

                    for l in current.keys():

                        current[l] = p.apply_async(
                            _get_arm_update, (self.arms[l], data))

                    for (l, r) in current.items():
                        l_losses = unzip(r.get())[1]
                        losses[l] += sum(l_losses)
                        self.num_pulls[l] += 1

                    k += 1

            sigma = sorted(
                losses.items(), 
                key=lambda x: x[1]) 
            comparator_j = int(num_arms_i / self.eta)

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

def _get_arm_update(arm, data):

    return arm.update(data)
