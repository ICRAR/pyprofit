#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
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

import argparse
import collections
import itertools
import math
import random
import sys
import time

import pyprofit


def powers_of_to_up_to(n):
    # Surely it's easier than this, but whatever...
    last_exponent = int(math.floor(math.log(n, 2)))
    powers = set([2**x for x in range(last_exponent + 1)] + [n])
    powers = list(powers)
    powers.sort()
    return powers

parser = argparse.ArgumentParser('')
parser.add_argument('-n', '--niter', help='Number of iterations, defaults to 100',
                    type=int, default=100)
parser.add_argument('-f', '--fft', help='Maximum FFT effort to test, defaults to 2',
                    type=int, default=2)
parser.add_argument('-t', '--omp_threads', help='Maximum OpenMP threads to use with FFT convolvers, defaults to 1',
                    type=int, default=1)
parser.add_argument('-r', '--reuse_psf', help='Measure PSF reusage (in FFT convolvers). Defaults to no',
                    action='store_true', default=False)

args = parser.parse_args()
n_iter = args.niter
max_fft_effort = args.fft
# We test with OpenMP threads [1, 2, 4, 8, ... N]
# If N is not a power of 2 it doesn't matter
omp_threads = powers_of_to_up_to(args.omp_threads)
reuse = (False, True) if args.reuse_psf else (False,)



print("Benchmark measuring with %d iterations" % (n_iter,))

# What we use to time iterative executions
timing_result = collections.namedtuple('timing_result', 't error')
def time_me(**kwargs):
    try:
        start = time.time()
        for _ in range(n_iter):
            pyprofit.make_model(kwargs)
        t = time.time() - start

        return timing_result(t / n_iter, None)
    except pyprofit.error as e:
        return timing_result(0, e)

# The profile and image/kernel sizes used throughout the benchmark
# We use a sky profile because it takes virtually no time to run
img_sizes= (100, 150, 200, 300, 400, 800)
krn_sizes = (25, 50, 100, 200)
profiles = {'sky': [{'bg': 10e-6}]}

# Initialize the kernels for all sizes with random data
krns = {}
for krn_size in krn_sizes:
    krns[krn_size] = [[random.random() for _ in range(krn_size)] for _ in range(krn_size)]

# Get the OpenCL platforms/devices information
cl_info = pyprofit.opencl_info()
def all_cl_devs():
    return ((p, d, cl_info[p][2][d][1]) for p in range(len(cl_info)) for d in range(len(cl_info[p][2])))

# Print OpenCL information onto the screen
print("OpenCL platforms/devices information:")
for plat, dev, has_double_support in all_cl_devs():
    print("[%s] %s / %s. Double: %s" % (
        '%d%d' % (plat, dev),
        cl_info[plat][0], cl_info[plat][2][dev][0],
        "Yes" if cl_info[plat][2][dev][1] else "No"))
print('')

# Get an float OpenCL environment for each of them
# If the device supports double, get a double OpenCL environment as well
sys.stdout.write('Getting an OpenCL environment for each of them now...')
sys.stdout.flush()
openclenvs = []
for p, dev, double_support in all_cl_devs():
    openclenvs.append(pyprofit.openclenv(p, dev, False))
    if double_support:
        openclenvs.append(pyprofit.openclenv(p, dev, True))
print(' done!')

# Build up the title and display it
title = "Img Krn     NoConv   BruteOld"
for omp_t in omp_threads:
    title += ' %10s' % ('Brute_%d' % (omp_t,))
for e, omp_t, r in itertools.product(range(max_fft_effort + 1), omp_threads, reuse):
    title += ' %10s' % ('FFT_%d_%d_%s' % (e, omp_t, "Y" if r else "N"))
for plat, dev, has_double_support in all_cl_devs():
    title += ' %10s' % ('cl_%d%d_f' % (plat, dev))
    title += ' %10s' % ('Lcl_%d%d_f' % (plat, dev))
    if has_double_support:
        title += ' %10s' % ('cl_%d%d_d' % (plat, dev))
        title += ' %10s' % ('Lcl_%d%d_d' % (plat, dev))
print('\n' + title)


def create_convolver(name, **kwargs):
    if sys.stdout.isatty():
        sys.stdout.write('Creating %s convolver...' % (name,))
        sys.stdout.flush()
    conv = pyprofit.make_convolver(**kwargs)
    if sys.stdout.isatty():
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
    return conv

# Main benchmarking process
errors = []
for img_size, krn_size in itertools.product(img_sizes, krn_sizes):

    # We don't test these
    if krn_size > img_size:
        continue

    # Create all required convolvers
    oldbrute_convolver = create_convolver('brute force (old)', width=img_size, height=img_size, psf=krns[krn_size], convolver_type='brute-old')

    # New brute convolvers with OpenMP threads
    brute_convolvers = []
    for omp_t in omp_threads:
        label = 'brute force (omp_threads = %d)' % (omp_t,)
        conv = create_convolver(label, width=img_size, height=img_size, psf=krns[krn_size],
                                convolver_type='brute', omp_threads=omp_t)
        brute_convolvers.append(conv)

    # FFT convolvers use different efforts and OpenMP thread
    fft_convolvers = []
    for e, omp_t, r in itertools.product(range(max_fft_effort + 1), omp_threads, reuse):
        label = 'FFT (effort = %d, omp_threads = %d, reuse = %s)' % (e, omp_t, "Yes" if r else "No")
        conv = create_convolver(label, width=img_size, height=img_size, psf=krns[krn_size],
                                convolver_type='fft', fft_effort=e, omp_threads=omp_t,
                                reuse_psf_fft=r)
        fft_convolvers.append(conv)

    # cl_convolvers include normal and local CL convolvers, in the correct order
    cl_convolvers = []
    for clenv in openclenvs:
        cl_convolvers.append(create_convolver('OpenCL', width=img_size, height=img_size, psf=krns[krn_size], convolver_type='opencl', openclenv=clenv))
        cl_convolvers.append(create_convolver('OpenCL (local)', width=img_size, height=img_size, psf=krns[krn_size], convolver_type='opencl-local', openclenv=clenv))

    # Basic profile calculation time
    profiles['sky'][0]['convolve'] = False
    t_profile = time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size])
    profiles['sky'][0]['convolve'] = True

    # ... and convolve with each of them!
    times = []
    for conv in [oldbrute_convolver] + brute_convolvers + fft_convolvers + cl_convolvers:
        times.append(time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size], convolver=conv))

    # Format results and print to the screen
    # If there was an error we print an "E%d" placeholder, which we expand
    # at the end
    fmt = "%d %-3d"
    args = [img_size, krn_size]

    all_times = [t_profile] + times
    for t in all_times:
        if t.error is None:
            fmt += " %10.4f"
            args.append(t.t)
        else:
            errno = "[E%d]" % len(errors)
            fmt += " %10s"
            args.append(errno)
            errors.append(t.error)

    print(fmt % tuple(args))

# Print all errors
if errors:
    print("\nError list:")
    for i, e in enumerate(errors):
        print("  E%d: %s" % (i, str(e)))