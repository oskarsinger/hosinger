from classes cimport PyRandomSvd as PyRSvd, PyEigenMatrixXd as PEM
from libcpp.vector cimport vector

from pygen import PygenMatrix

cdef class RandomSvd:
    cdef PyRSvd *r_svd

    def __cinit__(self):
        self.r_svd = new PyRSvd()

    def __dealloc(self):
        if self.r_svd is not NULL:
            del self.r_svd
