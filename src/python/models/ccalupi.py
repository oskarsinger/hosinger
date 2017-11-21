import numpy as np

# TODO: deal with N_o neq N_p
# TODO: consider having a CCA model instead of individual phi models
class CCAPenalizedLUPIModel:

    def __init__(self, 
        o_model,
        p_model,
        phi_o,
        phi_p,
        lambda_s,
        lambda_p,
        idn=None):

        self.o_model = o_model
        self.p_model = p_model
        self.phi_o = phi_o
        self.phi_p = phi_p
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.idn = idn

    def get_prediction(self, data):
        
        projected = self.phi_o.get_prediction(
            data, params[1])

        return self.o_model.get_prediction(
            projected, params[0])

    def get_gradient(self, data, params):

        # Compute projected stuff
        proj_o = self.phi_o.get_prediction(
            data[0], params[2]) 
        proj_p = self.phi_p.get_prediction(
            data[1], params[3]) 
        proj_diff = proj_o - proj_p

        # Compute w grads
        w_o_grad = self.o_model.get_gradient(
            proj_o,
            (params[0], params[2]))
        w_p_grad = self.p_model.get_gradient(
            proj_p, params[1])

        # Compute phi grads
        phi_o_grad = self.phi_o.get_gradient(
            data[0], (params[2], params[3]))
        phi_p_grad = self.phi_p.get_gradient(
            data[1], (params[3], params[2]))
        phi_o_f_grad = self.o_model.get_data_gradient(
            proj_o, params[0])
        phi_p_f_grad = self.p_model.get_data_gradient(
            proj_p, params[1])
        phi_o_ell_grad = np.dot(
            phi_o_grad.T, 
            phi_o_f_grad + self.lambda_s * proj_diff)
        phi_p_ell_grad = np.dot(
            phi_p_grad.T,
            self.lambda_p * phi_p_f_grad - self.lambda_s * proj_diff)

        return (
            w_o_grad,
            w_p_grad,
            phi_o_ell_grad,
            phi_p_ell_grad)

    def get_objective(self, data, params):

        # Compute projected stuff
        proj_o = self.phi_o.get_prediction(
            data[0], params[2]) 
        proj_p = self.phi_p.get_prediction(
            data[1], params[3]) 
        proj_diff = proj_o - proj_p

        # Compute individual objectives
        o_obj = self.o_model.get_objective(
            proj_o, params[0])
        p_obj = self.p_model.get_objective(
            proj_p, params[1])
        cca_obj = np.linalg.norm(proj_diff)**2 / 2

        return o_obj + lambda_s * cca_obj + lambda_p * p_obj

    def get_residuals(self, data, params):

        # Get CCA projections
        proj_o = self.phi_o.get_prediction(
            data[0], params[2]) 
        proj_p = self.phi_p.get_prediction(
            data[1], params[3]) 

        # Get prediction residuals
        o_residuals = self.o_model.get_residuals(
            proj_o, params[0])
        p_residuals = self.p_model.get_residuals(
            proj_p, params[1])

        # Get CCA residuals
        s_residuals = proj_o - proj_p

        return (
            o_residuals,
            p_residuals,
            s_residuals)

    def get_datum(self, data, i):

        (X_o, X_p, y) = data

        return (X_o[i,:], X_p[i,:], y[i,:])

    def get_projection(self, data, params):

        (o_pars, p_pars, phi_o_pars, phi_p_pars) = params
        projected_o = self.o_model.get_projected(
            data[0], o_pars)
        projected_p = self.p_model.get_projected(
            data[1], p_pars)
        projected_phi_o = self.phi_o.get_projected(
            data[0], phi_o_pars)
        projected_phi_p = self.phi_p.get_projected(
            data[1], phi_p_pars)

        return (
            projected_o,
            projected_p,
            projected_phi_o,
            projected_phi_p)

    def get_parameter_shape(self):

        o_shape = self.o_model.parameter_shape()
        p_shape = self.p_model.parameter_shape()
        phi_o_shape = self.phi_o.parameter_shape()
        phi_p_shape = self.phi_p.parameter_shape()

        return (
            o_shape,
            p_shape,
            phi_o_shape,
            phi_p_shape)
