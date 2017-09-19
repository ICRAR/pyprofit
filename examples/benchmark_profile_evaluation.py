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
parser.add_argument('-w', '--width', help='Image width',
                    type=int, default=200)
parser.add_argument('-H', '--height', help='Image height',
                    type=int, default=200)
parser.add_argument('-t', '--omp_threads', help='Maximum OpenMP threads to use for profile evaluation, defaults to 1',
                    type=int, default=1)

args = parser.parse_args()
n_iter = args.niter
width = args.width
height = args.height
omp_threads = powers_of_to_up_to(args.omp_threads)

print("Benchmark measuring profile image of %d x %d with %d iterations" % (width, height, n_iter,))

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

# All the combinations we'll try out. This makes out 10000 combinations,
# which we evaluate niter times

# nser in range [1, 8]
num_nsers = 10
nsers = [(x * 7. / (num_nsers - 1)) + 1 for x in range(num_nsers)]

# nser in range [0, 90]
num_angs = 5
angs = [x * 90. / (num_angs - 1) for x in range(num_angs)]

# axrat in range (0, 1]
num_axrats = 2
axrats = [(x + 1) / 10. for x in range(num_axrats)]

# re in range (0, width]
res = [(x + 1) * 10. for x in range(2)]

sersic_profile = {'xcen': width/2, 'ycen': height/2, 'mag': 10, 'rough': 0, 'convolve': False}
profiles = {'sersic': [sersic_profile]}

# Evaluate with an empty OpenCL environment (i.e., use CPU evaluation),
# then each of the OpenCL environments in turn, and then with different
# OpenMP threads
labels = ['CPU']
for p, dev, double_support in all_cl_devs():
    labels.append('CL_%d%d_f' % (p, dev))
    if double_support:
        labels.append('CL_%d%d_d' % (p, dev))
labels += ['OMP_%d' % t for t in omp_threads]

eval_args = [{}]
eval_args += [{'openclenv': clenv} for clenv in openclenvs]
eval_args += [{'omp_threads': t} for t in omp_threads]

parameters = (zip(labels, eval_args), nsers, angs, axrats, res)
times = collections.defaultdict(list)
for label_and_evalargs, nser, ang, axrat, re in itertools.product(*parameters):
    sersic_profile['nser'] = nser
    sersic_profile['ang'] = ang
    sersic_profile['axrat'] = axrat
    sersic_profile['re'] = re

    label, args = label_and_evalargs
    times[label].append(time_me(width=width, height=height, profiles=profiles, **args))

# Print values and exit
errors = []
def value(x):
    if x.error:
        ret = "[E%d]" % len(errors)
        errors.append(x.error)
        return "%8s" % ret
    return "%8.4f" % x.t

print(" ".join(["%8s" % (l) for l in labels]))
for vals in zip(*[times[x] for x in labels]):
    print(" ".join(value(x) for x in vals))

if errors:
    print("\nErrors founds:")
    for i, e in enumerate(errors):
        print("  E%d: %s" % (i, str(e)))