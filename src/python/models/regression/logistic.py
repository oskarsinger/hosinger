import numpy as np


class BinaryL2RegularizedLogisticRegressionModel:


    def __init__(self, p, gamma, idn=None):

        self.p = p
        self.gamma = gamma
        self.idn = idn


    def get_prediction(self, data, params):

        x = self._get_unpacked_data(data)[0]
        inside_exp = - np.dot(x, params)
        denom = 1 + np.exp(inside_exp)
        probs = np.power(denom, -1)

        return (probs > 0.5).astype(float)


    def get_gradient(self, data, params):

        (X, y, c) = self._get_unpacked_data(data)
        denom = np.exp(self.get_residuals(data, params))
        factor = - y * np.power(denom, -1) * c
        data_term = np.mean(factor * X, axis=0)
        regularization = self.gamma * params

        return data_term + regularization


    # TODO: this interface is kinda weird; maybe rethink it
    # TODO: make sure I don't need to norm by batch size anywhere
    # TODO: account for weighted individual loss functions
    def get_data_gradient(self, data, params, transform_grad):

        (X, y) = data
        denom = self.get_residuals(data, params)
        factor = - y * np.power(denom, -1)
        # TODO: the dimensions here get a little funky; double check them
        transform_term = np.dot(transform_grad, params.T)

        return np.sum(factor * transform_term, axis=0)


    def get_objective(self, data, params):

        (_, y, c) = self.get_unpacked_data(data)
        residuals = c * self.get_residuals(data, params)
        data_term = np.mean(residuals)
        param_norm = np.linalg.norm(params)**2
        regularization = self.gamma * param_norm / 2

        return data_term + regularization
        
        
    def get_residuals(self, data, params):

        (X, y) = self._get_unpacked_data(data)[:2]
        inside_exp = - y * np.dot(X, params)

        return np.log(1 + np.exp(inside_exp))


    def get_datum(self, data, i):

        (X, y, c) = self._get_unpacked_data(data)
        x_i = X[i,:][np.newaxis,:]
        y_i = y[i,:][np.newaxis,:]
        c_i = None if len(data) == 2 else c[i,:][np.newaxis,:]
        datum = (
            (x_i, y_i, c_i)
            if c_i else
            (x_i, y_i)
        )

        return datum


    def get_projected(self, data, params):

        return params


    def get_parameter_shape(self):

        return (self.p, 1)


    def _get_unpacked_data(self, data):

        (X, y, c) = [None] * 3

        if type(data) is list:
            num_data = len(data[0])
            X = np.array(
                datum[0] for datum in data]
            )

            if num_data > 1:
                y = np.array(
                    datum[1] for datum in data]
                )

                if num_data == 3:
                    c = np.array(
                        datum[2] for datum in data]
                    )
                else:
                    c = np.ones_like(y)
        else:
            if len(data) == 1:
                X = data
            elif len(data) == 2:
                (X, y) = data
                c = np.ones_like(data)
            else:
                (X, y, c) = data

        return (X, y ,c)
