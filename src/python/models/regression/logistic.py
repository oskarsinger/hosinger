import numpy as np

class BinaryL2RegularizedLogisticRegressionModel:

    def __init__(self, p, gamma, idn=None):

        self.p = p
        self.gamma = gamma
        self.idn = idn

    def get_gradient(self, data, params):

        (X, y) = data
        denom = self.get_residuals(data, params)
        factor = - y * np.power(denom, -1)
        data_term = np.mean(factor * X, axis=0)
        regularization = self.gamma * params

        return data_term + regularization

    # TODO: this interface is kinda weird; maybe rethink it
    # TODO: make sure I don't need to norm by batch size anywhere
    def get_data_gradient(self, data, params, transform_grad):

        (X, y) = data
        denom = self.get_residuals(data, params)
        factor = - y * np.power(denom, -1)
        # TODO: the dimensions here get a little funky; double check them
        transform_term = np.dot(transform_grad, params.T)

        return np.sum(factor * transform_term, axis=0)

    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)
        data_term = np.mean(np.log(residuals))
        param_norm = np.linalg.norm(params)**2
        regularization = self.gamma * param_norm / 2

        return data_term + regularization
        
    def get_residuals(self, data, params):

        (X, y) = data
        inside_exp = - y * np.dot(X, params)

        return 1 + np.exp(inside_exp)

    def get_datum(self, data, i):

        (A, b) = data
        a_i = A[i,:][np.newaxis,:]
        b_i = b[i,:][np.newaxis,:]

        return (a_i, b_i)

    def get_projected(self, data, params):

        return params

    def get_parameter_shape(self):

        return (self.p, 1)
