import numpy as np

class LinearRegressionGaussianLoader:

    def __init__(self,
        inner_loader,
        w=None,
        noise=None,
        noisy=False,
        bias=False):

        self.loader = inner_loader
        self.noise = noise
        self.noisy = noisy
        self.bias = bias

        self.X = self.loader.get_data()
        (self.n, self.p) = self.X.shape

        if bias:
            self.X = np.hstack(
                [self.X, np.ones((self.n, 1))])
            self.p += 1

        if w is None:
            w = np.random.randn(self.p, 1)
        elif self.bias:
            np.random.randn
            w = np.hstack

        self.w = w
        self.y = np.dot(self.X, self.w)
        self.noise = None

        if self.noise is not None:
            self.y += self.noise
        elif self.noisy:
            self.noise = np.random.randn(self.n, 1)
            self.y += self.noise

    def get_data(self):

        return (self.X, self.y)

    def name(self):

        return 'LinearRegressionGaussianLoader'

    def cols(self):

        return self.loader.cols()

    def rows(self):

        return self.loader.rows()
