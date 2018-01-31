import numpy as np

from fitterhappier.stepsize import InversePowerScheduler as IPS
from theline.utils import get_thresholded

# WARNING: all Pegasos implementations expect external step size of 1

class PegasosHingeLossSVMPlusWithSlacksModel:

    def __init__(self, 
        d_o,
        d_p,
        gamma=0.5, 
        C=10,
        theta=1, 
        i=None):

        self.d_o = d_o
        self.d_p = d_p
        self.d = self.d_o + self.d_p
        self.gamma = gamma
        self.C = C
        self.theta = theta
        self.lam_o = 1.0 / (self.C * self.theta)
        self.lam_p = self.gamma * self.lam_o
        self.i = i

        self.num_rounds = 0
        self.eta_scheduler_o = IPS(
            initial=self.lam_o**(-1), power=1)
        self.eta_scheduler_p = IPS(
            initial=self.lam_p**(-1), power=1)

    def get_gradient(self, data, params):

        self.num_rounds += 1
        
        # Extract data and params, get step size, etc.
        eta_o = self.eta_scheduler_o.get_stepsize()
        eta_p = self.eta_scheduler_p.get_stepsize()
        (X_o, X_p, y) = data
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]
        k = X.shape[0]

        # Compute hinge loss terms
        y_hat_mag = np.dot(X_o, w_o)
        y_prod = y * y_hat_mag
        X_p_prod = np.dot(X_p, w_p)
        y_and_X_p_prods = y_prod + X_p_prod
        lt_one = y_and_X_p_prods < 1

        # Compute w_o gradient
        data_o_term = eta_o * np.sum(
            y[lt_one,:] * X_o[lt_one,:],
            axis=0) / k
        reg_o_term = self.lam_o * eta_o * w_o
        w_o_grad = reg_o_term - data_o_term

        # Compute w_p gradient
        data_p_term1 = eta_p * np.sum(
            X_p[lt_one,:],
            axis=0) / k
        data_p_term2 = eta_p * np.sum(
            X_p[X_p_prod > 0,:],
            axis=0) / (self.theta * k)
        reg_p_term = self.lam_p * eta_p * w_p
        w_p_grad = reg_p_term - data_p_term1 - data_p_term2

        return np.vstack([w_o_grad, w_p_grad])

    def get_objective(self, data, params):

        # Extract parameters
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]

        # Computer regularization terms
        reg_o_term = self.lam_o * np.linalg.norm(w_o)**2 / 2
        reg_p_term = self.lam_p * np.linalg.norm(w_p)**2 / 2

        # Compute residual terms
        residuals = get_residuals(data, params)   
        data_term = residuals / residuals.shape[0]

        return reg_o_term + reg_p_term + data_term

    def get_residuals(self, data, params):

        # Extract parameters and data
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]
        (X_o, X_p, y) = data

        # Compute first hinge term residuals
        y_hat_mag = np.dot(X_o, w_o)
        y_prod = y * y_hat 
        X_p_prod = np.dot(X_p, w_p)
        y_and_X_p_prods = y_prod + X_p_prod
        threshd1 = get_thresholded(
            1 - y_and_X_p_prods, lower=0)

        # Compute second hinge term residuals
        threshd2 = get_thresholded(
            X_p_prod, lower=0)

        return threshd1 + threshed2 / self.theta

    def get_datum(self, data, i):

        (X_o, X_p, y) = data
        X_o_i = X_o[i,:][np.newaxis,:]
        X_p_i = X_p[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return (X_o_i, X_p_i, y_i)

    def get_projection(self, data, params):

        # Extract params
        w_o = params[:self.d_o,:]
        w_p = params[self.d_o:,:]

        # Extract scales
        scale_o = (np.linalg.norm(w_o) * np.sqrt(self.lam_o))**(-1)
        scale_p = (np.linalg.norm(w_p) * np.sqrt(self.lam_p))**(-1)

        # Project w_o and w_p
        w_o_projected = min([1, scale_o]) * w_o
        w_p_projected = min([1, scale_p]) * w_p

        return np.vstack([w_o_projected, w_p_projected])

# TODO: double check all this
class PegasosHingeLossWeightedLinearSVMModel:

    def __init__(self, p, i=None, lam=10**(-1)):

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
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag
        data_term = eta * np.sum(
            c * y[y_prod < 1] * X[y_prod < 1],
            axis=0) / k
        reg_term = eta * self.lam * params

        return reg_term - data_term

    def get_objective(self, data, params):

        residuals = get_residuals(data, params)   
        data_term = np.sum(residuals) / residuals.shape[0]
        reg_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return reg_term + data_term

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

        norm = np.linalg.norm(params)
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
        data_term = eta * np.sum(
            y[y_prod < 1] * X[y_prod < 1],
            axis=0) / k
        reg_term = eta * self.lam * params

        return reg_term - data_term

    def get_objective(self, data, params):

        residuals = get_residuals(data, params)   
        data_term = np.sum(residuals) / residuals.shape[0]
        reg_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return reg_term + data_term

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

        norm = np.linalg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)
