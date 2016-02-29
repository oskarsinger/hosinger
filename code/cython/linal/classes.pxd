cdef extern from "../../linal/random_matrix_factory.h" namespace "linal::random":
    cdef cppclass RandomMatrixFactory:
        RandomMatrixFactory()

cdef extern from "../../linal/py_eigen.h" namespace "linal::python":
    cdef cppclass PyEigen:
        PyEigen(int rows, int cols)
        double Get(int rows, int cols)
        void Set(int rows, int cols, double val)
        int Rows()
        int Cols()
