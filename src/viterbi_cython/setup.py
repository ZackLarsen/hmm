from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('viterbi_cython.pyx'))