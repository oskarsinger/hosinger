from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("rmf", 
        ["linal/rmf.pyx", "../linal/random_matrix_factory.cc"],
        include_dirs=["../eigen/", "../linal/"],
        extra_compile_args=["-std=c++11"],
        language="c++")
]

setup(
    ext_modules = cythonize(extensions),
)
