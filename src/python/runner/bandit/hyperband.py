from successive_halving import FiniteSuccessiveHalvingRunner as FSHR

class FiniteHyperBandRunner:

    def __init__(self,
        factory, 
        servers,
        arg_names,
        arg_ranges,
        max_rounds):

        self.factory = factory
        self.servers = servers
        self.arg_names = arg_names
        self.arg_ranges = arg_ranges
        self.max_rounds = max_rounds

        self.num_rounds = 0
        self.num_pulls = []
        self.models = []

    def run(self):

        

    def get_status(self):

        return {
            'factory': self.factory,
            'servers': self.servers,
            'arg_names': self.arg_names,
            'arg_ranges': self.arg_ranges,
            'num_rounds': self.num_rounds,
            'models': self.models,
            'num_pulls': self.num_pulls}
