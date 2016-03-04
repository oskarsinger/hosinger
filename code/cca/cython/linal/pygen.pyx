from classes cimport PyEigenMatrixXd as PEM

from numpy import array, zeros

cdef class PygenMatrix:
    cdef PEM *pem

    def __cinit__(self, numpy_mat=None):

        if numpy_mat is not None:
            (n,p) = numpy_mat.shape

            self.pem = new PEM(n,p)

            for i in range(n):
                for j in range(p):
                    self.pem.Set(i, j, numpy_mat[i,j])
        else:
            self.pem = NULL

    def __dealloc__(self):
        if self.pem is not NULL:
            del self.pem

    cdef from_pem(self, PEM *pem):

        self.pem = pem

    def to_numpy(self):

        cdef n = self.pem.Rows() 
        cdef p = self.pem.Cols()

        numpy_mat = zeros((n,p))

        for i in range(n):
            for j in range(p):
                numpy_mat[i,j] = self.pem.Get(i,j)

        return numpy_mat
