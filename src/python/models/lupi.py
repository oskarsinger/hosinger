import numpy as np

class CCALUPI:

    def __init__(self, 
        f_o, 
        f_p, 
        phi_o, 
        phi_p, 
        lambda_s,
        lambda_p,
        idn):

        self.f_o = f_o
        self.f_p = f_p
        self.phi_o = phi_o
        self.phi_p = phi_p
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.idn = idn

    def get_prediction(self, data):
        
        y_hat = None

        if type(data) is tuple:
            (data_o, data_p) = data
            y_hat_o = self.get_o_prediction(
                data[0], 
                (params[0], params[2]))
            y_hat_p = self.get_p_prediction(
                data[1], 
                (params[1], params[3]))
            y_hat = (y_hat_o, y_hat_p)
        else:
          - y_hat = self.get_o_prediction(
                data, params)

        return y_hat

    def get_o_prediction(self, data, params):

        projected_o = self.phi_o.get_prediction(
            data, params[1])

        return self.f_o.get_prediction(
            projected_o, params[0])

    def get_p_prediction(self, data, params):

        projected_p = self.phi_p.get_prediction(
            data, params[1])

        return self.f_p.get_prediction(
            projected_p, params[0])

    def get_gradient(self, data, params):

        # Compute projected stuff
        proj_o = self.phi_o.get_prediction(
            data[0], params[2]) 
        proj_p = self.phi_p.get_prediction(
            data[1], params[3]) 
        proj_diff = proj_o - proj_p

        # Compute w grads
        w_o_grad = self.f_o.get_gradient(
            proj_o,
            (params[0], params[2]))
        w_p_grad = self.f_p.get_gradient(
            proj_p, params[1])

        # Compute phi grads
        phi_o_grad = self.phi_o.get_gradient(
            data[0], params[2])
        phi_p_grad = self.phi_p.get_gradient(
            data[1], params[3])
        phi_o_f_grad = self.f_o.get_data_gradient(
            proj_o, params[0])
        phi_p_f_grad = self.f_p.get_data_gradient(
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

        pass

    def get_residuals(self, data, params):

        pass

    def get_datum(self, data, i):

        pass

    def get_projection(self, data, params):

        pass

    def get_parameter_shape(self):

        pass
