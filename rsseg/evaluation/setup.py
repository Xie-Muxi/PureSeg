"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

package = Extension(
    'bbox', ['box_overlaps.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
