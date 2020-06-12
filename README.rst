pyprofit
########

.. image:: https://travis-ci.org/ICRAR/pyprofit.svg?branch=master
    :target: https://travis-ci.org/ICRAR/pyprofit

.. image:: https://img.shields.io/pypi/v/pyprofit.svg
    :target: https://pypi.python.org/pypi/pyprofit

.. image:: https://img.shields.io/pypi/pyversions/pyprofit.svg
    :target: https://pypi.python.org/pypi/pyprofit

*pyprofit* is a python wrapper for `libprofit <https://www.github.com/ICRAR/libprofit>`_.

Installing
==========

*pyprofit* is available in `PyPI <https://pypi.python.org/pypi/pyprofit>`_
and thus can be easily installed via::

 pip install pyprofit

Since version 1.8.1 pre-compiled binary versions of the package
are available for Linux distributions.
These are offered for convenience
and come with OpenMP and FFTW support,
but lack features that some users may want to use,
like OpenCL, threaded-FFTW
and some CPU-specific optimizations.

Compiling
---------

If you need to *compile* this package
(either because a binary is not available in your platform,
or because you want to get the last drop of performance)
you will need to compile **and install** *libprofit* first.
For instruction on how to compile and install *libprofit* please read
`libprofit's documentation <http://libprofit.readthedocs.io/en/latest/getting.html#compiling>`_.

If you need to point to a specific libprofit installation
set the ``LIBPROFIT_HOME`` environment variable to point to it,
e.g.::

 LIBPROFIT_HOME=/opt/software/libprofit/1.8.0/ pip install pyprofit

Note that most users will need to specify
a user-writable installation directory
using the ``-DCMAKE_INSTALL_PREFIX=/my/installation/directory`` option
when invoking ``cmake`` as part of libprofit's compilation.
This is the same directory that then needs to be specified to pyprofit
via its ``LIBPROFIT_HOME`` environment variable.

All in all, the process of compiling both libprofit and pyprofit
should work similar to this::

 $> git clone https://github.com/ICRAR/libprofit
 $> mkdir libprofit/build
 $> cd libprofit/build
 $> cmake .. -DCMAKE_INSTALL_PREFIX=/my/libprofit/installation/directory
 $> make
 $> make install
 $> LIBPROFIT_HOME=/my/libprofit/installation/directory pip install pyprofit


Troubleshooting
---------------

If you are experiencing problems with your installation
(specially if you are compiling this package)
try the following::

 DISTUTILS_DEBUG=1 PYPROFIT_NO_MUTE=1 python setup.py -v build
