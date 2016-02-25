cdef extern from "random_matrix_factory.h":
    cdef cppclass RandomMatrixFactory:
        RandomMatrixFactory()

cdef class RandomMatrixFactory:
    cdef RandomMatrixFactory* thisptr

    def __cinit__(self):
        self.thisptr = new RandomMatrixFactory()

    def __dealloc__(self):
        del self.thisptr
