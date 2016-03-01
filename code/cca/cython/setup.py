from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

compile_args = [
    "-std=c++11"
]

linal_include_dirs = [
    "../eigen/",
    "../linal/"
]

extensions = [
    Extension("rmf", 
        ["linal/rmf.pyx", "../linal/random_matrix_factory.cc"],
        include_dirs=linal_include_dirs,
        extra_compile_args=compile_args,
        language="c++"),
    Extension("pygen",
        ["linal/pygen.pyx", "../linal/py_eigen.cc"],
        include_dirs=linal_include_dirs,
        extra_compile_args=compile_args,
        language="c++"),
    Extension("random_svd",
        ["linal/pyrsvd.pyx", "../linal/py_random_svd.cc"],
        include_dirs=linal_include_dirs,
        extra_compile_args=compile_args,
        language="c++")
]

setup(
    ext_modules = cythonize(extensions),
)
