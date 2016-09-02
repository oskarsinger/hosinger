from successive_halving import FiniteSuccessiveHalvingRunner as FSHR
from math import floor, log

class FiniteHyperBandRunner:

    def __init__(self,
        get_arms,
        servers,
        max_rounds,
        max_size,
        min_size,
        eta=3):

        self.get_arms = get_arms
        self.servers = servers
        self.max_rounds = max_rounds
        self.max_size = max_size
        self.min_size = min_size
        self.eta = eta

        self.num_rounds = 0
        self.num_pulls = []
        self.arms = []
        self.history = []

    def run(self):

        while self.num_rounds <= self.max_rounds:
            B = self.max_size * 2**(self.num_rounds)
            get_second = lambda x,b: int(floor(log(x,b)))
            size_ratio = float(self.max_size)/self.min_size
            num_rounds = min([
                B/self.max_size - 1,
                get_second(size_ratio, self.eta)])

            for l in xrange(num_rounds):
                ratio1 = B/self.max_size
                ratio2 = float(self.eta**(l)) / (l+1)
                n = int(floor(ratio1 * ratio2))
                arms = self.get_arms(n)
                sh = FSHR(
                    arms, 
                    servers, 
                    B, 
                    self.max_size, 
                    self.min_size,
                    eta=self.eta)

                sh.run()

                sh_info = sh.get_status()
                winner = arms[sh_info['winner']]
                loss = sh_info['winner_loss']
                num_pulls = sh_info['num_pulls']
                
                self.history.append((winner,loss))

                # TODO: figure out how to update num_pulls
                # TODO: in general make sure this is ready

            self.num_rounds += 1

    def get_status(self):

        return {
            'get_arms': self.get_arms,
            'servers': self.servers,
            'arg_names': self.arg_names,
            'arg_ranges': self.arg_ranges,
            'num_rounds': self.num_rounds,
            'arms': self.arms,
            'num_pulls': self.num_pulls}

class InfiniteHyperBandRunner:

    def __init__(self):

        'poop'

    def run(self):

        'poop'
