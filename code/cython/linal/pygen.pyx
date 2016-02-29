from classes cimport PyEigen as PE
from numpy import array, zeros

cdef class PygenMatrix:
    cdef PE *pe

    def __cinit__(self, mat):

        (rows, cols) = mat.shape

        self.pe = new PE(int(rows), int(cols))

        self._from_numpy(mat)

    def __dealloc__(self):
        if self.pe is not NULL:
            del self.pe

    def _from_numpy(self, mat):

        (rows, cols) = mat.shape

        cdef int i, j

        for i in xrange(rows):
            for j in xrange(cols):
                self.pe.Set(i, j, mat[i,j])

    def to_numpy(self):

        cdef int rows = self.rows()
        cdef int cols = self.cols()
        cdef int i,j
        A = zeros((rows, cols))

        for i in xrange(rows):
            for j in xrange(cols):
                A[i,j] = self.pe.Get(i, j)

        return A

    def get(self, int row, int col):
        
        return self.pe.Get(row, col)

    def set(self, int row, int col, double val):

        self.pe.Set(row, col, val)

    def rows(self):

        return self.pe.Rows()

    def cols(self):

        return self.pe.Cols()
