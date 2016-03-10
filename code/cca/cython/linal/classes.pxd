from libcpp.vector cimport vector

cdef extern from "../../linal/py_random_svd.h" namespace "linal::python":
    cdef cppclass PyRandomSvd:
        vector[vector[double]] U
        vector[vector[double]] s
        vector[vector[double]] V
        PyRandomSvd(const vector[vector[double]] &initial)
        void GetRandomSvd()
        void GetRandomSvd(int k)
        void GetRandomSvd(int k, int q)
