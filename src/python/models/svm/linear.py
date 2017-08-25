import numpy as np

from fitterhappier.stepsize import InversePowerScheduler as IPS
from linal.utils import get_thresholded

class PegasosHingeLossSVMModel:

    def __init__(self, p, i, lam=10**(-5)):

        self.p = p
        self.id_number = i
        self.lam = lam

        self.num_rounds = 0
        self.eta_scheduler = IPS(
            initial=self.lam**(-1), power=1)

    def get_gradient(self, data, params):

        self.num_rounds += 1

        eta = self.eta_scheduler.get_stepsize()
        X = data[0]
        y = data[1]
        k = X.shape[0]
        y_hat = np.dot(X, params)
        y_prod = y * y_hat
        data_term = np.sum(
            y[y_prod < 1] * X[y_prod < 1],
            axis=0) / * (self.lam * k)
        w_term = self.lam * params

        return w_term - data_term

    def get_objective(self, data, params):

        residuals = get_residuals(data, params)   
        r_term = np.sum(residuals) / residuals.shape[0]
        w_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return w_term + r_term

    def get_residuals(self, data, params):

        (X, y) = data
        y_hat = np.dot(A, params)
        y_prod = y * y_hat 
        threshd = get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        (A, b) = data
        a_i = A[i,:][np.newaxis,:]
        b_i = b[i,:][np.newaxis,:]

        return (a_i, b_i)

    def get_projection(self, data, params):

        norm = np.linalg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)
