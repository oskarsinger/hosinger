from classes cimport PyEigenMatrixXd as PEM

from numpy import array, zeros

cdef class PygenMatrix:
    cdef PEM *pem

    def __cinit__(self):

        self.pem = NULL

    def __dealloc__(self):
        if self.pem is not NULL:
            del self.pem
