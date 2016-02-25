from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    ext_modules=[
        Extension("random_matrix_factory",
            sources=["../linal/random_matrix_factory.pyx"],
            libraries=["librmf.so"],
            extra_link_args=["-L../shared/"],
            language="c++"),
        ],
    cmdclass = {'build_ext': build_ext},

)

