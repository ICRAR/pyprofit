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
import distutils
import distutils.ccompiler
import distutils.errors
import glob
import os
import sys
import tempfile

from setuptools import setup, Extension


# Check how C++11 has to be specified
def get_cpp11_stdspec():

    if 'PYPROFIT_CXX11' in os.environ:
        return os.environ['PYPROFIT_CXX11']

    code = """
    #include <iostream>
    int main() {
        for(auto i: {0,1,2}) {
            std::cout << i << std::endl;
        }
    }
    """
    source_fname = tempfile.mktemp(suffix='.cpp')
    with open(source_fname, 'w') as f:
        f.write(code)

    stdspec = None
    for stdspec_ in ['-std=c++11', '-std=c++0x', None]:
        try:
            compiler = distutils.ccompiler.new_compiler()
            object_fnames = compiler.compile([source_fname], extra_postargs=[stdspec_] if stdspec_ else [])
            stdspec = stdspec_
            os.remove(object_fnames[0])
            break
        except distutils.errors.CompileError:
            continue
    os.remove(source_fname)
    return stdspec

stdspec = get_cpp11_stdspec()
if stdspec is None:
    print("No C/C++ compiler with C++11 support found. "
          "Use the CC environment variable to specify a different compiler if you have one")
    sys.exit(1)
print("Using %s to enable C++11 support" % (stdspec,))

def has_system_gsl():
    compiler = distutils.ccompiler.new_compiler()
    return compiler.has_function('gsl_sf_gamma', libraries=['gsl', 'gslcblas'])

# Our module
pyprofit_sources = ['pyprofit.cpp']

# libprofit sources
pyprofit_sources += glob.glob('libprofit/src/*.cpp')

# include dirs
incdirs = ['libprofit']

# gsl sources
libs = []
if 'PYPROFIT_USE_BUNDLED_GSL' not in os.environ and has_system_gsl():
    print("")
    print("")
    print("Found GSL installation on your system, will link this module against it")
    print("")
    print("")
    libs += ['gsl', 'gslcblas']
else:
    print("")
    print("")
    print("Compiling module with bundled mini-GSL code")
    print("")
    print("")
    pyprofit_sources += glob.glob('gsl/specfunc/*.cpp')
    pyprofit_sources += glob.glob('gsl/cdf/*.cpp')
    pyprofit_sources += glob.glob('gsl/complex/*.cpp')
    pyprofit_sources += glob.glob('gsl/randist/*.cpp')
    pyprofit_sources += glob.glob('gsl/err/*.cpp')
    pyprofit_sources += glob.glob('gsl/sys/*.cpp')
    incdirs.append('gsl')


pyprofit_ext = Extension('pyprofit',
                       depends=glob.glob('libprofit/include/*.h'),
                       language='c++',
                       define_macros = [('HAVE_GSL',1)],
                       sources = pyprofit_sources,
                       include_dirs = incdirs,
                       libraries = libs,
                       extra_compile_args=[stdspec])

setup(
      name='pyprofit',
      version='0.13.0',
      description='Libprofit wrapper for Python',
      author='Rodrigo Tobar',
      author_email='rtobar@icrar.org',
      url='https://github.com/rtobar/pyprofit',
      classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
          "Operating System :: OS Independent",
          "Programming Language :: C++",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Topic :: Scientific/Engineering :: Astronomy"
      ],
      ext_modules = [pyprofit_ext]
)
