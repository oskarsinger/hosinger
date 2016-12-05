import numpy as np

class GaussianMixtureModel:

    def __init__(self, num_components):

        self.p = 3 * num_components

    def get_gradient(self, data, params):

        print 'Poop'

    def get_projection(self, data, params):

        print 'Poop'

    def get_error(self, data, params):

        print 'Poop'

    def get_parameter_shape(self):

        return (self.p, 1)
