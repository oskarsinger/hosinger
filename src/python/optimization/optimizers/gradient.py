from optimization.optimizers import AbstractOptimizer

class GradientOptimizer(AbstractOptimizer):

    def get_update(self, parameters, gradient, eta):

        return parameters - eta * gradient

    def get_status(self):

        return {}
