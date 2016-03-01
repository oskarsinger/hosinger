from classes cimport PyEigenMatrixXd as PEM
from numpy import array, zeros

cdef class PygenMatrix:
    cdef PEM *pem

    def __cinit__(self):

        self.pem = NULL

    def __dealloc__(self):
        if self.pem is not NULL:
            del self.pem

    def from_numpy(self, mat):

        (rows, cols) = mat.shape
        cdef PEM *pem = new PEM(rows, cols)

        cdef int i, j

        for i in xrange(rows):
            for j in xrange(cols):
                pem.Set(i, j, mat[i,j])

        return PygenMatrix_Init(pem)

    def to_numpy(self):

        cdef int rows = self.rows()
        cdef int cols = self.cols()
        cdef int i,j
        A = zeros((rows, cols))

        for i in xrange(rows):
            for j in xrange(cols):
                A[i,j] = self.pem.Get(i, j)

        return A

cdef PygenMatrix_Init(PEM *pem):
    mat = PygenMatrix()
    mat.pem = pem

    return mat
