class InverseSquareRootScheduler:

    def __init__(self, initial):

        self.initial = initial

    def get_stepsize(self, num_rounds):

        denom = float(num_rounds)**(-0.5)

        return denom * self.initial
