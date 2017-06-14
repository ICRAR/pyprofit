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
import subprocess
import sys
import tempfile

from setuptools import setup, Extension
import setuptools
from setuptools.command.build_ext import build_ext


def enrich_with_ldflags(compiler):
    if os.name != 'posix':
        return
    if 'LDFLAGS' not in os.environ or not os.environ['LDFLAGS']:
        return
    dirs = filter(lambda x: x.strip(), os.environ['LDFLAGS'].split('-L'))
    for d in dirs:
        compiler.add_library_dir(d)

class mute_compiler(object):

    def __init__(self):
        self.devnull = self.oldstderr = None

    def __enter__(self):
        if 'PYPROFIT_NO_MUTE' not in os.environ and os.name == 'posix':
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
        enrich_with_ldflags(c)
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
        enrich_with_ldflags(c)
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

def has_openmp():

    # User doesn't want OpenMP support
    if 'PYPROFIT_NO_OPENMP' in os.environ:
        return None

    code = """
    #include <omp.h>
    int main() {
    #ifdef _OPENMP
      return 0;
    #else
      breaks_on_purpose
    #endif
    }
    """

    # This should cover most compilers; otherwise we can always add more...
    for flag in ("", "-fopenmp", "-fopenmp=libomp", "-openmp", "-xopenmp", "-mp"):
        distutils.log.debug("-- Trying OpenMP support with %s", flag)
        if compiles(code, extra_postargs=[flag]):
            distutils.log.info("-- OpenMP support available via %s", flag)
            return flag

    return None

def has_fftw():

    # User doesn't want FFTW support
    if 'PYPROFIT_NO_FFTW' in os.environ:
        return False

    # Check the headers are actually there
    code = "#include <fftw3.h>\nint main() {}"
    if not compiles(code):
        distutils.log.error("-- FFTW headers not found")
        return False
    distutils.log.debug("-- FFTW headers found")

    # Check the library can be linked
    with mute_compiler() as c:
        enrich_with_ldflags(c)
        has_func = c.has_function('fftw_cleanup', libraries=['fftw3'])
    distutils.log.debug("-- FFTW library %s", "found" if has_func else "not found")
    return has_func

def has_fftw_omp(openmp_flag):

    # Check the library can be linked
    with mute_compiler() as c:
        enrich_with_ldflags(c)
        has_func = c.has_function('fftw_plan_with_nthreads', libraries=['fftw3', 'fftw3_omp'])
    distutils.log.debug("-- FFTW OpenMP library %s", "found" if has_func else "not found")
    return has_func

class configure(setuptools.Command):
    """Configure command to enrich the pyprofit extension"""

    def initialize_options(self):
        pass
    finalize_options = initialize_options
    description = 'Configures the pyprofit extension'
    user_options = []

    def _generate_config_h(self, defs):

        distutils.log.info("-- Generating config.h")

        this_dir = os.path.dirname(__file__)
        version_file = os.path.join(this_dir, 'libprofit', 'VERSION')
        config_in_file = os.path.join(this_dir, 'libprofit', 'profit', 'config.h.in')
        config_h_file = os.path.join(this_dir, 'libprofit', 'profit', 'config.h')

        with open(version_file, 'rt') as f:
            version = f.read().strip()

        # The following items need to be replaced:
        #  * #cmakedefine PROFIT_{USES_{R,GSL},DEBUG,OPENMP,OPENCL,FFTW{,_OPENMP}}
        #  * #define PROFIT_VERSION "@PROFIT_VERSION@"
        #  * #define PROFIT_OPENCL_MAJOR @PROFIT_OPENCL_MAJOR@
        #  * #define PROFIT_OPENCL_MINOR @PROFIT_OPENCL_MINOR@
        cmakedefines = [('PROFIT_USES_R', False),
                        ('PROFIT_USES_GSL', True),
                        ('PROFIT_DEBUG', False)]
        for macro in ('PROFIT_OPENMP', 'PROFIT_OPENCL', 'PROFIT_FFTW', 'PROFIT_FFTW_OPENMP'):
            cmakedefines.append((macro, macro in defs))

        defines = [('PROFIT_VERSION', version),
                   ('PROFIT_OPENCL_MAJOR', defs.get('PROFIT_OPENCL_MAJOR', '')),
                   ('PROFIT_OPENCL_MINOR', defs.get('PROFIT_OPENCL_MINOR', ''))]

        replacements = []
        for k, v in defines:
            replacements.append('s/@%s@/%s/' % (k, v))
        for k, v in cmakedefines:
            replacements.append('s/^#cmakedefine %s/#%s %s/' % (k, 'define' if v else 'undef', k))
        replacements = '; '.join(replacements)

        cmd = ['sed', replacements, config_in_file]
        with open(config_h_file, 'wb') as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False)
            output, _ = p.communicate()
            retcode = p.poll()
            if retcode:
                raise Exception("Error while running %s: %s" % (cmd, output))
            f.write(output)

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
        defines = {'HAVE_GSL': 1}
        extra_compile_args=[stdspec] if stdspec else []
        extra_link_args = []

        # Optional OpenCL support
        if has_opencl():
            maj,min = max_opencl_ver()
            distutils.log.info("-- Compiling pyprofit with OpenCL %d.%d support", maj, min)
            libs.append('OpenCL')
            defines['PROFIT_OPENCL'] = 1
            defines['PROFIT_OPENCL_MAJOR'] = maj
            defines['PROFIT_OPENCL_MINOR'] = min
        else:
            distutils.log.info("-- Compiling pyprofit without OpenCL support")

        # Optional OpenMP support
        openmp_flag = has_openmp()
        if openmp_flag is not None:
            defines['PROFIT_OPENMP'] = 1
            extra_compile_args.append(openmp_flag)
            extra_link_args.append(openmp_flag)
        else:
            distutils.log.info("-- No OpenMP support available")

        # Optional FFTW support
        if has_fftw():
            defines['PROFIT_FFTW'] = 1
            libs.append('fftw3')
            if openmp_flag and has_fftw_omp(openmp_flag):
                defines['PROFIT_FFTW_OPENMP'] = 1
                libs.append('fftw3_omp')
                distutils.log.info("-- Compiling pyprofit with OpenMP-enabled FFTW support")
            else:
                distutils.log.info("-- Compiling pyprofit with FFTW support")
        else:
            distutils.log.info("-- Compiling pyprofit without FFTW support")

        self._generate_config_h(defines)

        pyprofit_ext.libraries = libs
        pyprofit_ext.define_macros = [('PROFIT_BUILD', 1)]
        pyprofit_ext.extra_compile_args = extra_compile_args
        pyprofit_ext.extra_link_args = extra_link_args

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
      version='1.4.1',
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
          "Programming Language :: Python :: 3.6",
          "Topic :: Scientific/Engineering :: Astronomy"
      ],
      ext_modules = [pyprofit_ext],
      cmdclass = {
        'configure': configure,
        'build_ext': _build_ext,
      }
)
