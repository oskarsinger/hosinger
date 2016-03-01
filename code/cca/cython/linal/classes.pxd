from libcpp.vector cimport vector

cdef extern from "../../linal/random_matrix_factory.h" namespace "linal::random":
    cdef cppclass RandomMatrixFactory:
        RandomMatrixFactory()

cdef extern from "../../linal/py_eigen_matrix.h" namespace "linal::python":
    cdef cppclass PyEigenMatrixXd:
        PyEigenMatrixXd(int rows, int cols)
        double Get(int rows, int cols)
        void Set(int rows, int cols, double val)
        int Rows()
        int Cols()

cdef extern from "../../linal/py_random_svd.h" namespace "linal::python":
    cdef cppclass PyRandomSvd:
        PyRandomSvd()
        vector[PyEigenMatrixXd] GetRandomSvd(PyEigenMatrixXd A)
        vector[PyEigenMatrixXd] GetRandomSvd(PyEigenMatrixXd A, int k)
        vector[PyEigenMatrixXd] GetRandomSvd(PyEigenMatrixXd A, int k, int q)
