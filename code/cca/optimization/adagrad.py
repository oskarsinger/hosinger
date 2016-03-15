import numpy as np

from linal import utils

class AdaGrad:

    def __init__(self, objective, gradient, projection, eta, delta):

        self._objective = objective
        self._gradient = gradient
        self._projection = projection
        self._eta = eta
        self._delta = delta

    def update(self, minibatch):

        print "Some stuff"
