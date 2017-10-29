import numpy as np

from fitterhappier.stepsize import InversePowerScheduler as IPS
from theline.utils import get_thresholded

# TODO: double check all this
class PegasosHingeLossWeightedSVMModel:

    def __init__(self, p, i=None, lam=10**(-5)):

        self.p = p
        self.idn = i
        self.lam = lam

        self.num_rounds = 0
        self.eta_scheduler = IPS(
            initial=self.lam**(-1), power=1)

    def get_gradient(self, data, params):

        self.num_rounds += 1

        eta = self.eta_scheduler.get_stepsize()
        ((X, c), y) = data
        k = X.shape[0]
        y_hat = np.dot(X, params)
        y_prod = y * y_hat
        data_term = np.sum(
            c * y[y_prod < 1] * X[y_prod < 1],
            axis=0) / (self.lam * k)
        w_term = self.lam * params

        return w_term - data_term

    def get_objective(self, data, params):

        residuals = get_residuals(data, params)   
        r_term = np.sum(residuals) / residuals.shape[0]
        w_term = np.thelineg.norm(params)**2 * self.lam / 2
        
        return w_term + r_term

    def get_residuals(self, data, params):

        ((X, c), y) = data
        y_hat = np.dot(X, params)
        y_prod = y * y_hat 
        threshd = c * get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        ((X, c), y) = data
        x_i = X[i,:][np.newaxis,:]
        c_i = c[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return ((x_i, c_i), y_i)

    def get_projection(self, data, params):

        norm = np.thelineg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)

class PegasosHingeLossSVMModel:

    def __init__(self, p, i=None, lam=10**(-5)):

        self.p = p
        self.idn = i
        self.lam = lam

        self.num_rounds = 0
        self.eta_scheduler = IPS(
            initial=self.lam**(-1), power=1)

    def get_gradient(self, data, params):

        self.num_rounds += 1

        eta = self.eta_scheduler.get_stepsize()
        (X, y) = data
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
        w_term = np.thelineg.norm(params)**2 * self.lam / 2
        
        return w_term + r_term

    def get_residuals(self, data, params):

        (X, y) = data
        y_hat = np.dot(X, params)
        y_prod = y * y_hat 
        threshd = get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        (X, y) = data
        x_i = X[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return (x_i, y_i)

    def get_projection(self, data, params):

        norm = np.thelineg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)
