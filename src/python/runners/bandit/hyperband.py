from successive_halving import FiniteSuccessiveHalvingRunner as FSHR
from drrobert.misc import unzip
from math import floor, log, ceil

# TODO: so far this is specialized to AppGrad a bit; fix that later
class FiniteHyperBandRunner:

    def __init__(self,
        get_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size,
        eta=3):

        self.get_arm = get_arm
        self.ds_list = ds_list
        self.num_views = len(ds_list)
        self.max_size = max_size
        self.min_size = min_size
        self.eta = eta

        self.s_max = int(0.9*log(self.max_size, self.eta))
        self.B = (self.s_max + 1)*self.max_size
        self.num_pulls = []
        self.arms = []
        self.history = []

    def run(self):

        print 'Running HyperBand for', self.s_max+1, 'rounds'

        print_interval = (self.s_max+1) / 10

        for s in reversed(range(self.s_max+1)):

            i = self.s_max + 1 - s
            should_print = i % print_interval == 0

            if should_print:
                print 'HyperBand Round', i

            num_arms = int(ceil(
                self.B/self.max_size/s*self.eta**s))
            num_rounds = self.max_size*self.eta**(-s)

            if should_print:
                print 'Generating', num_arms, 'arms'
            
            (arms, parameters) = unzip(
                [self.get_arm()
                 for i in xrange(num_arms)])

            if should_print:
                print 'Initializing SuccessiveHalvingRunner'

            sh = FSHR(
                arms, 
                self.ds_list, 
                s+1,
                num_rounds,
                self.max_size, 
                self.min_size,
                eta=self.eta)

            if should_print:
                print 'Running SuccessiveHalvingRunner'

            sh.run()

            sh_info = sh.get_status()
            windex = sh_info['winner']
            winner = arms[windex]
            winning_parameters = parameters[windex]
            loss = sh_info['winner_loss']
            num_pulls = sh_info['num_pulls']
            
            self.history.append(
                (winner,winning_parameters, loss))

            # TODO: figure out how to update num_pulls
            # The indexing is weird

            if should_print:
                print 'Refreshing data servers'

            # TODO: should I be doing this?
            for ds in self.ds_list:
                ds.refresh()

    def get_status(self):

        return {
            'get_arms': self.get_arms,
            'ds_list': self.ds_list,
            'arms': self.arms,
            'num_pulls': self.num_pulls,
            'history': self.history}

class InfiniteHyperBandRunner:

    def __init__(self):

        'poop'

    def run(self):

        'poop'