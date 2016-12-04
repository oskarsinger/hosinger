import numpy as np

class GaussianLinearRegression:

    def __init__(self, p):

        self.p = p

    def get_gradient(self, data, params):

        (A, b) = data
        b_hat = np.dot(A, params)
        residuals = b_hat - b

        return np.dot(A.T, residuals)

    def get_error(self, data, params):

        (A, b) = data
        b_hat = np.dot(A, params)
        residuals = b_hat - b

        return np.linalg.norm(residuals)

    def get_parameter_shape(self):

        return (self.p, 1)
