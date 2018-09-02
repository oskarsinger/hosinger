import numpy as np

from theline.utils import get_thresholded

# WARNING: all Pegasos implementations expect external step size of 1/t
class PegasosHingeLossLinearSVMPlusModel:

    def __init__(self, 
        dp, 
        do, 
        i=None, 
        gamma=10**(-1), 
        theta=10**(1), 
        C=10**(1)):

        self.dp = dp
        self.do = do
        self.idn = i
        self.gamma = gamma
        self.theta = theta
        self.C = C
        self.C_theta_2 = self.C * self.theta * 2

        self.num_rounds = 0

    def get_gradient(self, data, params):

        (X_o, X_p, y) = data
        (w_o, w_p) = params
        (res_o, res_p) = self.get_residuals(
            data,
            params)
        reg_o_term = 2 * w_o / self.C_theta_2
        data_o_term = np.mean(
            y[res_o > 0,:] * X_o[res_o > 0,:], 
            axis=0)
        grad_o = reg_o_term - data_o_term
        reg_p_term = 2 * self.gamma * w_p / self.C_theta_2
        data_p_term = np.mean(
            X_p[res_o > 0,:] + X_p[res_p > 0,:] / self.theta,
            axis=0)
        grad_p = reg_p_term - data_p_term

        return (grad_o, grad_p)

    def get_objective(self, data, params):

        (X_o, X_p, y) = data
        (w_o, w_p) = params
        reg_o_term = np.linalg.norm(w_o)**2 / self.C_theta_2
        reg_p_term = self.gamma * np.linalg.norm(w_p)**2 / self.C_theta_2
        reg_term = reg_o_term + reg_p_term
        (res_0, res_p) = self.get_residuals(
            data,
            params)
        data_o_term = np.mean(res_o)
        data_p_term = np.mean(res_p) / self.theta
        data_term = data_o_term + data_p_term

        return reg_term + data_term

    def get_residuals(self, data, params):

        (X_o, X_p, y) = data
        (w_o, w_p) = params
        y_term = y * np.dot(X_o, w_o)
        p_term = np.dot(X_p, w_p)
        res_o = np.maximum(0, 1 - y_term - p_term)
        res_p = np.maximum(0, - p_term)

        return (res_o, res_p)

    def get_datum(self, data, i):

        (X_o, X_p, y) = data

        return (X_o[i,:], X_p[i,:], y[i,:])

    def get_projected(self, data, params):

        constant_o = 2 / self.C_theta_2
        constant_p = self.gamma * 2 / self.C_theta_2
        (w_o, w_p) = params
        norm_o = np.linalg.norm(w_o)
        norm_p = np.linalg.norm(w_p)
        scale_o = (norm_o * np.sqrt(constant_o))**(-1)
        scale_p = (norm_p * np.sqrt(constant_p))**(-1)
        min_scale_o = min([1, scale_o])
        min_scale_p = min([1, scale_p])
        projected_w_o = min_scale_o * w_o
        projected_w_p = min_scale_p * w_p

        return (projected_w_o, projected_w_p)

    def get_parameter_shape(self):

        return ((self.dp, 1), (self.do, 1))

# TODO: double check all this
class PegasosHingeLossWeightedLinearSVMModel:

    def __init__(self, p, i=None, lam=10**(-1)):

        self.p = p
        self.idn = i
        self.lam = lam

        self.num_rounds = 0

    def get_gradient(self, data, params):

        self.num_rounds += 1

        ((X, c), y) = data
        k = X.shape[0]
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag
        data_term = np.sum(
            c * y[y_prod < 1] * X[y_prod < 1],
            axis=0) / (k * self.lam)

        return params - data_term

    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)   
        data_term = np.mean(residuals)
        reg_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return reg_term + data_term

    def get_residuals(self, data, params):

        ((X, c), y) = data
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag
        threshd = c * get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        ((X, c), y) = data
        x_i = X[i,:][np.newaxis,:]
        c_i = c[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return ((x_i, c_i), y_i)

    def get_projected(self, data, params):

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

    def get_gradient(self, data, params):

        self.num_rounds += 1

        (X, y) = data
        k = X.shape[0]
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag
        data_term = np.sum(
            y[y_prod < 1] * X[y_prod < 1],
            axis=0) / (k * self.lam)

        return params - data_term

    def get_objective(self, data, params):

        residuals = self.get_residuals(data, params)   
        data_term = np.sum(residuals) / residuals.shape[0]
        reg_term = np.linalg.norm(params)**2 * self.lam / 2
        
        return reg_term + data_term

    def get_residuals(self, data, params):

        (X, y) = data
        y_hat_mag = np.dot(X, params)
        y_prod = y * y_hat_mag 
        threshd = get_thresholded(
            1 - y_prod, lower=0)

        return threshd

    def get_datum(self, data, i):

        (X, y) = data
        x_i = X[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]

        return (x_i, y_i)

    def get_projected(self, data, params):

        norm = np.linalg.norm(params)
        scale = (norm * np.sqrt(self.lam))**(-1)
        min_scale = min([1, scale])

        return min_scale * params

    def get_parameter_shape(self):

        return (self.p, 1)
