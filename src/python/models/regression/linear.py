import numpy as np

class LinearRegressionModel:

    def __init__(self, p, idn=None):

        self.p = p
        self.idn = idn

    def get_gradient(self, data, params, batch=None):

        A = data[0]
        residuals = self.get_residuals(
            data, params, batch=batch)

        if batch is not None:
            A = A[:,batch]

            if np.isscalar(batch):
                A = A[:,np.newaxis]

        return np.dot(A.T, residuals)

    def get_objective(self, data, params):

        residuals = self.get_residuals(
            data, params)

        return np.linalg.norm(residuals)**2 / residuals.shape[0]

    def get_residuals(self, data, params, batch=None):

        (A, b) = data

        if batch is not None:
            A = A[:,batch]
            params = params[batch,:]

            if np.isscalar(batch):
                A = A[:,np.newaxis]
                params = params[:,np.newaxis]

        b_hat = np.dot(A, params)

        return b_hat - b

    def get_coordinate_sums(self, data):

        A = data[0]

        return np.sum(A, axis=0)

    def get_datum(self, data, i):

        (A, b) = data
        a_i = A[i,:][np.newaxis,:]
        b_i = b[i,:][np.newaxis,:]

        return (a_i, b_i)

    def get_projected(self, data, params):

        return params

    def get_parameter_shape(self):

        return (self.p, 1)
