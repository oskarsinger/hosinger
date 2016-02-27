cimport classes
from classes cimport RandomMatrixFactory as RMF

# distutils: language = c++
# distutils: sources = random_matrix_factory.cc

cdef class RandomMatrixFactory:
    cdef RMF *rmf

    def __cinit__(self):
        self.rmf = new RMF()

    def __dealloc__(self):
        if self.rmf is not NULL:
            del self.rmf
