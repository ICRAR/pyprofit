#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2016
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
from setuptools import setup, Extension

import os
import glob
from distutils import ccompiler

# For the time being we use GSL to satisfy our needs
# Does numpy come with something similar perhaps?
compiler = ccompiler.new_compiler()
if not compiler.has_function('gsl_sf_beta',libraries=('gsl','gslcblas')):
    raise Exception("")


pyprofit_sources = ['pyprofit.c']
pyprofit_sources += glob.glob('src/*.c')
pyprofit_ext = Extension('pyprofit',
                       sources = pyprofit_sources,
                       include_dirs = ['include'],
                       libraries=['gsl', 'gslcblas'])

setup(
      name='pyprofit',
      version='0.1.1',
      description='Libprofit wrapper for Python',
      author='Rodrigo Tobar',
      author_email='rtobar@icrar.org',
      url='https://github.com/rtobar/pyprofit',
      classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
          "Operating System :: OS Independent",
          "Programming Language :: C",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Astronomy"
      ],
      ext_modules = [pyprofit_ext]
)
