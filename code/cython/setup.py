from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    ext_modules=[
        Extension("random_matrix_factory",
            sources=["linal/random_matrix_factory.pyx", "random_matrix_factory.cpp"],
            include_dirs=["../eigen/", "../linal/"],
            language="c++"),
        ],
    cmdclass = {'build_ext': build_ext},

)

