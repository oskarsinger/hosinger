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

        (X1, X2, Y, C) = [None] * 4

        if len(data) == 4:
            (X1, X2, Y, C) = data 
        else:
            (X1, X2, Y) = data
            C = np.ones_like(y)

        denom = np.exp(self.get_residuals(data, params))
        factor = - Y * np.power(denom, -1) * C
        data_term = get_multi_dot([X2.T, factor, X1])
        regularization = self.gamma * params

        return data_term + regularization
        
        
    def get_objective(self, data, params):

        (Y, C) = [None] * 2

        if len(data) == 4:
            (_, _, Y, C) = data 
        else:
            (_, _, Y) = data
            C = np.ones_like(y)

        residuals = C * self.get_residuals(data, params)
        data_term = np.mean(residuals) / 2
        param_norm = np.linalg.norm(params)**2
        regularization = self.gamma * param_norm / 2

        return data_term + regularization


    def get_residuals(self, data, params):

        (X1, X2, Y) = data[:3]
        inside_exp = - Y * get_multi_dot([X1, params, X2.T])

        return np.log(1 + np.exp(inside_exp))


    def get_datum(self, data, i, j):

        (X1, X2, Y, C) = [None] * 4

        if len(data) == 4:
            (X1, X2, Y, C) = data
        else:
            (X1, X2, Y) = data

        x1_i = X1[i,:][np.newaxis,:]
        x2_j = X2[j,:][np.newaxis,:]
        y_ij = Y[i,j]
        c_ij = None if len(data) == 3 else C[i,j]

        datum = (
            (x1_i, x2_j, y_ij, c_ij) 
            if c_ij else 
            (x1_i, x2_j, y_ij)
        )
        
        return datum


    def get_projected(self, data, params):

        return params


    def get_parameter_shape(self);

        return (self.p, self.q)
