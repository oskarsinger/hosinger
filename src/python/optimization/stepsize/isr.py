class InverseSquareRootScheduler:

    def __init__(self, initial):

        self.initial = initial
        self.num_rounds = 0

    def get_stepsize(self):

        denom = float(self.num_rounds+1)**(-0.5)

        return denom * self.initial

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'initial': self.initial,
            'num_rounds': self.num_rounds}
