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

    def get_random_svd(self, numpy_mat, k=None):
        
        pygen_mat = PygenMatrix.from_numpy(numpy_mat)

        cdef PEM *pem = extract<PEM*>(pygen_mat.pem)
        cdef vector[PEM] c_svd 
        
        if k is None:
            c_svd = self.r_svd.GetRandomSvd(pem)
        else:
            c_svd = self.r_svd.GetRandomSvd(pem, k)

        py_svd = []

        for i in range(3):
            py_svd.append(PygenMatrix_Init(c_svd[i]).to_numpy())

        return py_svd
