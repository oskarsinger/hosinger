from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os

compile_args = [
    "-std=c++11"
]

eigen_dir = "../eigen/"
linal_dir = "../linal/"
linal_include_dirs = [
    eigen_dir,
    linal_dir
]

linal_sources = [
    "random_svd",
    "py_random_svd",
    "py_eigen_matrix",
    "random_matrix_factory",
    "random_orthonormal_basis"
]
linal_sources = [linal_dir + linal_source + ".cc"
                 for linal_source in linal_sources]

extensions = [
    Extension("pygen",
        ["linal/pygen.pyx"]+[linal_sources[2]],
        include_dirs=linal_include_dirs,
        extra_compile_args=compile_args,
        language="c++"),
    Extension("random_svd",
        ["linal/random_svd.pyx"]+linal_sources,
        include_dirs=linal_include_dirs,
        extra_compile_args=compile_args,
        language="c++")
]

setup(
    ext_modules = cythonize(extensions),
)
