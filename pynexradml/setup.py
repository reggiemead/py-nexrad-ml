from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "PyNEXRAD",
    ext_modules = cythonize("*.pyx"),
)
