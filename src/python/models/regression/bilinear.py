import numpy as np

from theline.utils import get_multi_dot

# TODO: account for sparse label matrix y with scipy sparse matrices
class BinaryL2RegularizedBilinearLogisticRegressionModel:


    def __init__(self, p, q, gamma, idn=None):

        (self.p, self.q) = (p, q)
        self.gamma = gamma
        self.idn = idn


    def get_prediction(self, data, params):

        (X1, X2) = data
        inside_exp = - get_multi_dot([X1, params, X2.T])
        denom = 1 + np.exp(inside_exp)
        probs = np.power(denom, -1)

        return (probs > 0.5).astype(float)


    def get_gradient(self, data, params):

        (X1, X2, y) = data
        denom = self.get_residuals(data, params)
        factor = - y * np.power(denom, -1)
        data_term = get_multi_dot([X2.T, factor, X1])
        regularization = self.gamma * params

        return data_term + regularization
        
        
    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)
        data_term = np.mean(np.log(residuals)) / 2
        param_norm = np.linalg.norm(params)**2
        regularization = self.gamma * param_norm / 2

        return data_term + regularization


    def get_residuals(self, data, params):

        (X1, X2, y) = data
        inside_exp = - y * get_multi_dot([X1, params, X2.T])

        return 1 + np.exp(inside_exp)


    def get_datum(self, data, i, j):

        (X1, X2, y) = data
        x1_i = X1[i,:][np.newaxis,:]
        x2_j = X2[j,:][np.newaxis,:]
        y_ij = y[i,j]

        return (x1_i, x2_j, y_ij)


    def get_projected(self, data, params):

        return params


    def get_parameter_shape(self);

        return (self.p, self.q)
