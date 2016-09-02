from drrobert.arithmetic import get_moving_avg as get_ma

class GradientOptimizer:

    def __init__(self, forget_factor=None):

        self.forget_factor = forget_factor
        self.alpha = 1
        self.beta = 1
        self.moving_avg = None

        if forget_factor is not None:
            self.alpha = forget_factor
            self.beta = 1 - self.alpha

    def get_update(self, parameters, gradient, eta):

        if self.moving_avg is None:
            self.moving_avg = gradient
        else:
            old = self.alpha * self.moving_avg
            new = self.beta * gradient
            self.moving_avg = old + new

        return parameters - eta * self.moving_avg

    def get_status(self):

        return {
            'forget_factor': self.forget_factor,
            'alpha': self.alpha,
            'beta': self.beta,
            'moving_avg': self.moving_avg}
