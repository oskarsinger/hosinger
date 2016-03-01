from classes cimport PyRandomSvd as PyRSvd, PyEigenMatrixXd as PEM
from libcpp.vector cimport vector

from pygen import PygenMatrix

cdef class RandomSvd:
    cdef PyRSvd *pr_svd

    def __cinit__(self):
        self.pr_svd = new PyRSvd()

    def __dealloc(self):
        if self.pr_svd is not NULL:
            del self.pr_svd

    def get_random_svd(PygenMatrix A):
        
        cdef PEM mat = A.get_matrix()
        cdef vector[PEM] svd = self.svd.GetRandomSvd(mat)
