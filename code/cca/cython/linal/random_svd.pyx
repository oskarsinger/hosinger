from classes cimport PyRandomSvd as PyRSvd
from libcpp.vector cimport vector

from pygen import PygenMatrix

from cython.operator cimport dereference as deref

cdef class RandomSvd:
    cdef PyRSvd *r_svd

    def __cinit__(self, numpy_mat):
        self.r_svd = new PyRSvd(numpy_mat.tolist())

    def __dealloc__(self):
        if self.r_svd is not NULL:
            del self.r_svd

    def get_svd(self, k=None):

        (n,p) = numpy_mat.shape

        if k is not None:
            self.r_svd.GetRandomSvd(k)
        else:
            self.r_svd.GetRandomSvd()

        U = self.r_svd.U
        s = self.r_svd.s
        V = self.r_svd.V

        return (U,s,V)
