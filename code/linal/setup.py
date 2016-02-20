from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name="Stuff",
    ext_modules=[
        Extension("random_matrix_factory",
            sources=["random_matrix_factory.pyx", "random_matrix_factory.o"],
            language="c++"),
        ],
    cmdclass = {'build_ext': build_ext},

)

