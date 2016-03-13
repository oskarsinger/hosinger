from classes cimport PyRandomSvd as PyRSvd
from libcpp.vector cimport vector

from numpy import array

cdef class RandomSvd:
    cdef PyRSvd *r_svd

    def __cinit__(self, numpy_mat):
        self.r_svd = new PyRSvd(numpy_mat.tolist())

    def __dealloc__(self):
        if self.r_svd is not NULL:
            del self.r_svd

    def get_svd(self, k=None):

        if k is not None:
            self.r_svd.GetRandomSvd(k)
        else:
            self.r_svd.GetRandomSvd()

        U = array(self.r_svd.U)
        s = array(self.r_svd.s)
        V = array(self.r_svd.V).T

        return (U,s,V)
