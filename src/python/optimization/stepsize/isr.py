class InversePowerScheduler:

    def __init__(self, initial=0.1, power=0.5):

        self.initial = initial
        self.num_rounds = 0
        self.power = power

    def get_stepsize(self):

        denom = float(self.num_rounds+1)**(-self.power)
        self.num_rounds += 1

        return denom * self.initial

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'initial': self.initial,
            'num_rounds': self.num_rounds}
