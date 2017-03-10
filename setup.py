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
from distutils.dep_util import newer_group
import distutils.errors
import glob
import os
import sys
import tempfile

from setuptools import setup, Extension
import setuptools
from setuptools.command.build_ext import build_ext


class mute_compiler(object):

    def __init__(self):
        self.devnull = self.oldstderr = None

    def __enter__(self):
        if os.name == 'posix':
            self.devnull = open(os.devnull, 'w')
            self.oldstderr = os.dup(sys.stderr.fileno())
            self.oldstdout = os.dup(sys.stdout.fileno())
            os.dup2(self.devnull.fileno(), sys.stderr.fileno())
            os.dup2(self.devnull.fileno(), sys.stdout.fileno())

        compiler = distutils.ccompiler.new_compiler()
        distutils.sysconfig.customize_compiler(compiler) # CC, CFLAGS, LDFLAGS, etc
        return compiler

    def __exit__(self, typ, err, st):
        if self.oldstderr is not None:
            os.dup2(self.oldstderr, sys.stderr.fileno())
            os.dup2(self.oldstdout, sys.stdout.fileno())
        if self.devnull is not None:
            self.devnull.close()

def compiles(code=None, source_fname=None, *args, **kwargs):

    if (code and source_fname) or (not code and not source_fname):
        raise ValueError("code XOR source_fname")

    if code:
        source_fname = tempfile.mktemp(suffix='.cpp')
        with open(source_fname, 'w') as f:
            f.write(code)

    devnull = oldstderr = None
    try:
        with mute_compiler() as c:
            object_fnames = c.compile([source_fname], *args, **kwargs)
        os.remove(object_fnames[0])
        return True
    except distutils.errors.CompileError:
        return False
    finally:
        # We wrote it, we take care of it
        if code:
            os.remove(source_fname)

def get_cpp11_stdspec():

    # User knows already how to do it
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
        distutils.log.debug("-- Trying compiler argument '%s' to enable C++11 support", stdspec_)
        extra_postargs=[stdspec_] if stdspec_ else []
        if compiles(source_fname=source_fname, extra_postargs=extra_postargs):
            stdspec = stdspec_
            break
    os.remove(source_fname)
    return stdspec

def has_gsl():

    # Check the headers are actually there
    code = "#include <gsl/gsl_sf_gamma.h>\nint main() {}"
    if not compiles(code):
        distutils.log.error("-- GSL headers not found")
        return False
    distutils.log.debug("-- GSL headers found")

    # Check the library can be linked
    with mute_compiler() as c:
        has_func = c.has_function('gsl_sf_gamma', libraries=['gsl', 'gslcblas'])
    distutils.log.debug("-- GSL library %s", "found" if has_func else "not found")
    return has_func

def opencl_include():
    return "OpenCL/opencl.h" if sys.platform == "darwin" else "CL/opencl.h"

def has_opencl():

    # User doesn't want OpenCL support
    if 'PYPROFIT_NO_OPENCL' in os.environ:
        return False

    # Check the headers are actually there
    code = """
        #include <%s>
        int main() {}
        """ % (opencl_include())
    if not compiles(code):
        distutils.log.info("-- OpenCL headers not found")
        return False
    distutils.log.debug("-- OpenCL headers found")

    # Check the library can be linked
    with mute_compiler() as c:
        has_func = c.has_function('clCreateContext', libraries=['OpenCL'])
    distutils.log.debug("-- OpenCL library %s", "found" if has_func else "not found")
    return has_func

def max_opencl_ver():

    inc = opencl_include()
    for ver in ((2,0), (1,2), (1,1), (1,0)):
        maj,min = ver
        distutils.log.debug("-- Looking for OpenCL %d.%d", maj, min)

        code = """
        #include <iostream>
        #include <%s>

        int main() {
            std::cout << CL_VERSION_%d_%d << std::endl;
        }
        """ % (inc, maj, min)
        if compiles(code):
            return maj,min

class configure(setuptools.Command):
    """Configure command to enrich the pyprofit extension"""

    def initialize_options(self):
        pass
    finalize_options = initialize_options
    description = 'Configures the pyprofit extension'
    user_options = []

    def run(self):

        # We should only be configuring the pyprofit module
        assert(len(self.distribution.ext_modules) == 1)
        pyprofit_ext = self.distribution.ext_modules[0]

        # Find (if any) the C++11 flag required to compile our code
        stdspec = get_cpp11_stdspec()
        if stdspec is None:
            msg = "\n\nNo C/C++ compiler with C++11 support found." + \
                  "Use the CC environment variable to specify a different compiler if you have one.\n" + \
                  "You can also try setting the PYPROFIT_CXX11 environment variable with the necessary switches " + \
                  "(e.g., PYPROFIT_CXX11='-std c++11')"
            raise distutils.errors.DistutilsPlatformError(msg)
        distutils.log.info("-- Using '%s' to enable C++11 support", stdspec)

        # Scan for GSL
        if not has_gsl():
            msg = "No GSL installation found on your system. " + \
                  "Install the GSL development package and try again\n\n"
            raise distutils.errors.DistutilsPlatformError(msg)
        distutils.log.info("-- Found GSL headers/lib")

        libs = ['gsl', 'gslcblas']
        defines = [('PROFIT_BUILD', 1), ('HAVE_GSL',1)]
        extra_compile_args=[stdspec] if stdspec else []

        # Optional OpenCL support
        if has_opencl():
            maj,min = max_opencl_ver()
            distutils.log.info("-- Compiling pyprofit with OpenCL %d.%d support", maj, min)
            libs.append('OpenCL')
            defines.append(('PROFIT_OPENCL',1))
            defines.append(('PROFIT_OPENCL_MAJOR', maj))
            defines.append(('PROFIT_OPENCL_MINOR', min))
        else:
            distutils.log.info("-- Compiling pyprofit without OpenCL support")

        pyprofit_ext.libraries = libs
        pyprofit_ext.define_macros = defines
        pyprofit_ext.extra_compile_args = extra_compile_args

class _build_ext(build_ext):
    """Custom build_ext command that includes the configure command"""

    def run(self):

        assert(len(self.distribution.ext_modules) == 1)
        ext = self.distribution.ext_modules[0]

        # Don't even run configure if not necessary
        ext_path = self.get_ext_fullpath(ext.name)
        depends = ext.sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            distutils.log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return

        self.run_command('configure')
        build_ext.run(self)


# The initial definition of the pyprofit module
# It is enriched during the 'configure' step
pyprofit_ext = Extension('pyprofit',
                       depends=glob.glob('libprofit/profit/*.h') + glob.glob('libprofit/profit/cl/*'),
                       language='c++',
                       sources = ['pyprofit.cpp'] + glob.glob('libprofit/src/*.cpp'),
                       include_dirs = ['libprofit'])

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
      ext_modules = [pyprofit_ext],
      cmdclass = {
        'configure': configure,
        'build_ext': _build_ext,
      }
)
