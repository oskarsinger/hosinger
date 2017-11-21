import numpy as np

# TODO: deal with N_o neq N_p
# TODO: consider having a CCA model instead of individual phi models
class CCAPenalizedLUPIModel:

    def __init__(self, 
        o_model,
        p_model,
        s_model,
        lambda_s,
        lambda_p,
        idn=None):

        self.o_model = o_model
        self.p_model = p_model
        self.s_model = s_model
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.idn = idn

    def get_prediction(self, data):
        
        projected = self.s_model.get_prediction(
            data[0], params[2])

        return self.o_model.get_prediction(
            projected, params[0])

    def get_gradient(self, data, params):

        # Compute projected stuff
        proj_o = self.s_model.get_prediction(
            data[0], params[2]) 
        proj_p = self.s_model.get_prediction(
            data[1], params[3]) 
        proj_diff = proj_o - proj_p

        # Compute w grads
        o_grad = self.o_model.get_gradient(
            proj_o,
            params[0])
        p_grad = self.p_model.get_gradient(
            proj_p, 
            params[1])

        # Compute phi grads
        (phi_o_grad, phi_p_grad) = self.s_model.get_gradient(
            data, 
            (params[2], params[3]))
        phi_o_f_grad = self.o_model.get_data_gradient(
            proj_o, 
            params[0],
            data[0].T)
        phi_p_f_grad = self.p_model.get_data_gradient(
            proj_p, 
            params[1],
            data[1].T)
        phi_o_ell_grad = np.dot(
            phi_o_grad.T, 
            phi_o_f_grad + self.lambda_s * proj_diff)
        phi_p_ell_grad = np.dot(
            phi_p_grad.T,
            self.lambda_p * phi_p_f_grad - self.lambda_s * proj_diff)
        s_grad = (phi_o_ell_grad, phi_p_ell_grad)

        return (
            o_grad,
            p_grad,
            s_grad)

    def get_objective(self, data, params):

        # Compute projected stuff
        proj_o = self.s_model.get_prediction(
            data[0], params[2]) 
        proj_p = self.s_model.get_prediction(
            data[1], params[3]) 
        proj_diff = proj_o - proj_p

        # Compute individual objectives
        o_obj = self.o_model.get_objective(
            proj_o, params[0])
        p_obj = self.p_model.get_objective(
            proj_p, params[1])
        s_obj = self.s_model.get_objective(
            data,
            (params[2], params[3]))

        return o_obj + lambda_s * s_obj + lambda_p * p_obj

    def get_residuals(self, data, params):

        # Get CCA projections
        proj_o = self.s_model.get_prediction(
            data[0], params[2]) 
        proj_p = self.s_model.get_prediction(
            data[1], params[3]) 

        # Get prediction residuals
        o_residuals = self.o_model.get_residuals(
            proj_o, params[0])
        p_residuals = self.p_model.get_residuals(
            proj_p, params[1])

        # Get CCA residuals
        s_residuals = self.s_model.get_residuals(
            data, (params[2], params[3]))

        return (
            o_residuals,
            p_residuals,
            s_residuals)

    def get_datum(self, data, i):

        (X_o, X_p, y) = data

        return (X_o[i,:], X_p[i,:], y[i,:])

    def get_projection(self, data, params):

        proj_o = self.o_model.get_projected(
            data[0], params[0])
        proj_p = self.p_model.get_projected(
            data[1], params[1])
        proj_s = self.s_model.get_projected(
            data, (params[2], params[3]))

        return (
            proj_o,
            proj_p,
            proj_s)

    def get_parameter_shape(self):

        o_shape = self.o_model.parameter_shape()
        p_shape = self.p_model.parameter_shape()
        s_shape = self.s_model.parameter_shape()

        return (
            o_shape,
            p_shape,
            s_shape)
