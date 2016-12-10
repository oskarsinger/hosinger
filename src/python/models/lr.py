import numpy as np

class LinearRegression:

    def __init__(self, p):

        self.p = p

    def get_gradient(self, data, params):

        A = data[0]
        residuals = self.get_residuals(
            data, params)

        return np.dot(A.T, residuals)

    def get_objective(self, data, params):

        residuals = self.get_residuals(
            data, params)

        return np.linalg.norm(residuals)

    def get_residuals(self, data, params):

        (A, b) = data
        b_hat = np.dot(A, params)

        return b_hat - b

    def get_coordinate_counts(self, data):

        A = data[0]

        return np.sum(
            (A != 0).astype(float),
            axis=0)

    def get_datum(self, data, i):

        (A, b) = data
        a_i = A[i,:][np.newaxis,:]
        b_i = b[i,:][np.newaxis,:]

        return (a_i, b_i)

    def get_projection(self, data, params):

        return params

    def get_parameter_shape(self):

        return (self.p, 1)
