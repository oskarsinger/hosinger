from classes cimport PyRandomSvd as PyRSvd, PyEigenMatrixXd as PEM
from libcpp.vector cimport vector

from pygen import PygenMatrix

from cython.operator cimport dereference as deref

cdef class RandomSvd:
    cdef PyRSvd *r_svd

    def __cinit__(self):
        self.r_svd = new PyRSvd()

    def __dealloc__(self):
        if self.r_svd is not NULL:
            del self.r_svd

    def get_svd(self, numpy_mat, k=None):

        (n,p) = numpy_mat.shape

        cdef PEM *pem = new PEM(n, p)

        for i in range(n):
            for j in range(p):
                self.pem.Set(i, j, numpy_mat[i,j])

        cdef vector[PEM] svd

        if k is not None:
            svd = self.r_svd.GetRandomSvd(deref(pem), k)
        else:
            svd = self.r_svd.GetRandomSvd(deref(pem))

        U = pem2numpy(&svd[0])
        s = pem2numpy(&svd[1])
        V = pem2numpy(&svd[2])

        return (U,s,V)

cdef pem2numpy(PEM *pem):

    temp = PygenMatrix()
    temp.from_pem(pem)
    
    return temp.to_numpy()
