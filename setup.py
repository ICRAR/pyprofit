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
import re
import subprocess
import sys
import tempfile

from setuptools import setup, Extension
import setuptools
from setuptools.command.build_ext import build_ext

# TODO: use an explicit temporary directory, and delete it at the end of
# everything


#
# Versions of libprofit against which this extension works
# Format is (major, minor, patch, suffix)
#
libprofit_versions = (
    (1, 7, 0, None),
    (1, 7, 1, None),
    (1, 7, 2, None),
    (1, 7, 3, None),
    (1, 7, 4, None),
    (1, 8, 0, 'dev'),
    (1, 8, 0, None),
    (1, 8, 1, None),
    (1, 8, 2, None),
)

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


def check_libprofit_version(h):

    major, minor, patch, suffix = -1, -1, -1, None
    with open(h, 'rt') as h:
        h = h.read()

        # libprofit < 1.6 defined PROFIT_VERSION only
        m = re.search(r'#define\WPROFIT_VERSION\W"(\d\.\d\.\d)"', h)
        if m:
            return tuple(map(int, m.group(1).split('.')))

        # libprofit >= 1.6 defines different macros for major, minor and patch version
        # libprofit >= 1.7 defines also a macro for a version suffix
        m = re.search(r'#define\WPROFIT_VERSION_MAJOR\W(\d)', h)
        if m:
            major = int(m.group(1))
        m = re.search(r'#define\WPROFIT_VERSION_MINOR\W(\d)', h)
        if m:
            minor = int(m.group(1))
        m = re.search(r'#define\WPROFIT_VERSION_PATCH\W(\d)', h)
        if m:
            patch = int(m.group(1))
        m = re.search(r'#define\WPROFIT_VERSION_SUFFIX\W"(\w+)"', h)
        if m:
            suffix = m.group(1)

    if major == -1 or minor == -1 or patch == -1:
        return None

    return major, minor, patch, suffix

def version_as_str(version):
    ver_str = '.'.join(map(str, version[:3]))
    if version[3] is not None:
        ver_str += '-' + version[3]
    return ver_str

def has_libprofit(user_incdirs, user_libdirs, extra_compile_args):

    # Manually check the headers are actually there, and that the version is
    # one we support
    builtin_dirs = ['/usr', '/usr/local']
    if sys.platform == "darwin":
        builtin_dirs += ['/opt/local']
    builtin_incdirs = [x + '/include' for x in builtin_dirs]
    builtin_libdirs = [x + '/lib' for x in builtin_dirs]
    builtin_libdirs += [x + '/lib64' for x in builtin_dirs]

    incdir = None
    found = []
    distutils.log.debug('-- Looking for libprofit headers under: %r', user_incdirs + builtin_incdirs)
    for i in user_incdirs + builtin_incdirs:
        header = os.path.join(i, 'profit', 'config.h')
        if not os.path.exists(header):
            distutils.log.debug('-- No libprofit headers under %s', header)
            continue
        distutils.log.debug("-- Found libprofit headers in %s, checking version" % header)
        version = check_libprofit_version(header)
        distutils.log.debug("-- Found libprofit version %s in %s" % (version_as_str(version), header))
        found.append((header, version))
        if version in libprofit_versions:
            distutils.log.info("-- Found libprofit headers for version %s", version_as_str(version))
            incdir = i
            break

    if not incdir:
        msg = "-- no suitable libprofit headers not found"
        if found:
            msg += '. The following headers were found though:\n'
            msg += '\n'.join('   %s (%s)' % (h, version_as_str(v)) for h, v in found)
            msg += '\n'
        distutils.log.error(msg)
        return None

    source_fname = tempfile.mktemp(suffix='.cpp')
    with open(source_fname, 'w') as f:
        f.write('int main() {}')

    # Check the library can be linked (and against which library dir)
    libdir = None
    with mute_compiler() as c:
        c.add_include_dir(incdir)
        c.add_library('profit')

        try:
            object_fnames = c.compile([source_fname], extra_preargs=extra_compile_args)
        except:
            return False

        for l in user_libdirs + builtin_libdirs:
            try:
                c.link_executable(object_fnames, 'test', library_dirs=[l])
                os.unlink('test')
                libdir = l
                break
            except:
                pass

    if not libdir:
        distutils.log.error("-- libprofit library not found")
        return None
    else:
        distutils.log.info("-- Found libprofit under %s", libdir)

    return incdir, libdir

class configure(setuptools.Command):
    """Configure command to enrich the pyprofit extension"""

    def initialize_options(self):
        pass
    finalize_options = initialize_options
    description = 'Configures the pyprofit extension'
    user_options = []

    def run(self):

        LIBPROFIT_HOME   = os.environ.get('LIBPROFIT_HOME', None)
        LIBPROFIT_INCDIR = os.environ.get('LIBPROFIT_INCDIR', None)
        LIBPROFIT_LIBDIR = os.environ.get('LIBPROFIT_LIBDIR', None)

        user_incdirs = []
        user_libdirs = []
        if LIBPROFIT_HOME:
            LIBPROFIT_HOME = os.path.expanduser(LIBPROFIT_HOME)
            user_incdirs.append(os.path.join(LIBPROFIT_HOME, 'include'))
            user_libdirs.append(os.path.join(LIBPROFIT_HOME, 'lib'))
            user_libdirs.append(os.path.join(LIBPROFIT_HOME, 'lib64'))
        if LIBPROFIT_INCDIR:
            LIBPROFIT_INCDIR = os.path.expanduser(LIBPROFIT_INCDIR)
            user_incdirs.append(LIBPROFIT_INCDIR)
        if LIBPROFIT_LIBDIR:
            LIBPROFIT_LIBDIR = os.path.expanduser(LIBPROFIT_LIBDIR)
            user_libdirs.append(LIBPROFIT_LIBDIR)

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
        extra_compile_args = [stdspec] if stdspec else []

        # Scan for libprofit
        info = has_libprofit(user_incdirs, user_libdirs, extra_compile_args)
        if not info:
            msg = ("No libprofit installation found on your system.\n\n"
                   "Supported versions are: %s\n\n"
                   "You can specify a libprofit installation directory via the LIBPROFIT_HOME "
                   "environment variable.\n"
                   "Additionally, you can also use the LIBPROFIT_INCDIR and LIBPROFIT_LIBDIR "
                   "environment variables\n"
                   "to point separately to the headers and library directories respectivelly\n\n"
                   "For example:\n\n"
                   "LIBPROFIT_HOME=~/local python setup.py install")
            msg = msg % ', '.join(version_as_str(v) for v in libprofit_versions)
            raise distutils.errors.DistutilsPlatformError(msg)
        distutils.log.info("-- Found libprofit headers/lib")

        pyprofit_ext.libraries = ['profit']
        pyprofit_ext.include_dirs = [info[0]]
        pyprofit_ext.library_dirs = [info[1]]
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
pyprofit_ext = Extension('pyprofit', language='c++', sources = ['pyprofit.cpp'])

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, 'README.rst'), 'rt') as f:
    long_description = f.read()

setup(
      name='pyprofit',
      version='1.8.2',
      description='Libprofit wrapper for Python',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Rodrigo Tobar',
      author_email='rtobar@icrar.org',
      url='https://github.com/ICRAR/pyprofit',
      classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
          "Operating System :: OS Independent",
          "Programming Language :: C++",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Topic :: Scientific/Engineering :: Astronomy"
      ],
      ext_modules = [pyprofit_ext],
      cmdclass = {
        'configure': configure,
        'build_ext': _build_ext,
      }
)
