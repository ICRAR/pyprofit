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

def new_compiler():
    compiler = distutils.ccompiler.new_compiler()
    distutils.sysconfig.customize_compiler(compiler) # CC, CFLAGS, LDFLAGS, etc
    return compiler

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
    for stdspec_ in ['-std=c++11', '-std=c++0x', '-std=c++', '']:
        try:
            compiler = new_compiler()
            object_fnames = compiler.compile([source_fname], extra_postargs=[stdspec_] if stdspec_ else [])
            stdspec = stdspec_
            os.remove(object_fnames[0])
            break
        except distutils.errors.CompileError:
            continue
    os.remove(source_fname)
    return stdspec

def has_gsl():
    compiler = new_compiler()
    return compiler.has_function('gsl_sf_gamma', libraries=['gsl', 'gslcblas'])

def opencl_include():
    return "OpenCL/opencl.h" if sys.platform == "darwin" else "CL/opencl.h"

def has_opencl():
    if 'PYPROFIT_NO_OPENCL' in os.environ:
        return False
    compiler = new_compiler()
    return compiler.has_function('clCreateContext', libraries=['OpenCL'])

def max_opencl_ver():

    compiler = new_compiler()
    inc = opencl_include()

    for ver in ((2,0), (1,2), (1,1), (1,0)):
        maj,min = ver

        code = """
        #include <iostream>
        #include <%s>

        int main() {
            std::cout << CL_VERSION_%d_%d << std::endl;
        }
        """ % (inc, maj, min)
        source_fname = tempfile.mktemp(suffix='.cpp')
        with open(source_fname, 'w') as f:
            f.write(code)

        try:
            compiler = new_compiler()
            object_fnames = compiler.compile([source_fname])
            os.remove(object_fnames[0])
            return ver
        except distutils.errors.CompileError:
            pass
        finally:
            os.remove(source_fname)

# Our module
pyprofit_sources = ['pyprofit.cpp']

# libprofit sources
pyprofit_sources += glob.glob('libprofit/src/*.cpp')

# include dirs
incdirs = ['libprofit']

defines = [('PROFIT_BUILD', 1)]

stdspec = get_cpp11_stdspec()
if stdspec is None:
    print("No C/C++ compiler with C++11 support found."
          "Use the CC environment variable to specify a different compiler if you have one.\n"
          "You can also try setting the PYPROFIT_CXX11 environment variable with the necessary switches "
          "(e.g., PYPROFIT_CXX11='-std c++11')")
    sys.exit(1)
print("Using '%s' to enable C++11 support" % (stdspec,))

# gsl libs
if not has_gsl():
    print("\n\nNo GSL installation found on your system. Install the GSL development package and try again\n\n")
    sys.exit(1)

libs = ['gsl', 'gslcblas']
defines.append(('HAVE_GSL',1))

# OpenCL support
if has_opencl():
    maj,min = max_opencl_ver()
    print("Compiling pyprofit with OpenCL %d.%d support" % (maj, min))
    libs.append('OpenCL')
    defines.append(('PROFIT_OPENCL',1))
    defines.append(('PROFIT_OPENCL_MAJOR', maj))
    defines.append(('PROFIT_OPENCL_MINOR', min))

pyprofit_ext = Extension('pyprofit',
                       depends=glob.glob('libprofit/profit/*.h'),
                       language='c++',
                       define_macros = defines,
                       sources = pyprofit_sources,
                       include_dirs = incdirs,
                       libraries = libs,
                       extra_compile_args=[stdspec] if stdspec else [])

setup(
      name='pyprofit',
      version='1.2.1',
      description='Libprofit wrapper for Python',
      author='Rodrigo Tobar',
      author_email='rtobar@icrar.org',
      url='https://github.com/ICRAR/pyprofit',
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
