from classes cimport PyEigenMatrixXd as PEM

from numpy import array, zeros

cdef class PygenMatrix:
    cdef PEM *pem

    def __cinit__(self):

        self.pem = NULL

    def __dealloc__(self):
        if self.pem is not NULL:
            del self.pem

    def from_numpy(self, numpy_mat):

        (n,p) = numpy_mat.shape

        cdef PEM *pem = new PEM(n,p)

        for i in range(n):
            for j in range(p):
                self.pem.set(i, j, numpy_mat[i,j])

        return PygenMatrix_Init(pem)

    def to_numpy(self):

        cdef n = self.pem.Rows() 
        cdef p = self.pem.Cols()

        numpy_mat = zeros((n,p))

        for i in range(n):
            for j in range(p):
                numpy_mat[i,j] = self.pem.Get(i,j)

        return numpy_mat

cdef PygenMatrix_Init(PEM *pem):
    mat = PygenMatrix()
    mat.pem = pem

    return mat
