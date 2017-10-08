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

import numpy as np
import pyprofit

def powers_of_to_up_to(n):
    # Surely it's easier than this, but whatever...
    last_exponent = int(math.floor(math.log(n, 2)))
    powers = set([2**x for x in range(1, last_exponent + 1)] + [n])
    powers = list(powers)
    powers.sort()
    if 1 in powers:
        powers.remove(1)
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
parser.add_argument('-N', '--nsers', help='Sersic indexes to sample, defaults to 1,8,10', default=None)
parser.add_argument('-a', '--angs', help='Angles to sample, defaults to 0,45,4', default=None)
parser.add_argument('-A', '--axrats', help='Axis ratios to sample, defaults to 0.1,1,4', default=None)
parser.add_argument('-r', '--res', help='Re values sample, defaults to 0,width/2,5', default=None)
parser.add_argument('-b', '--boxes', help='Boxing values to sample, defaults to 0.5,0.5,3', default=None)

args = parser.parse_args()
n_iter = args.niter
width = args.width
height = args.height
omp_threads = powers_of_to_up_to(args.omp_threads)


def define_parameter_range(name, spec):
    Min, Max, n = map(float, spec.split(','))
    diff = Max - Min
    step = diff/(n - 1)
    values = np.arange(Min, Max + step, step)

    # Avoid rounding errors
    if values[-1] != Max:
        values[-1] = Max

    print("%d %s: %r" % (n, name, values))
    return values

nsers = define_parameter_range('nser', args.nsers or '1,12,10')
angs = define_parameter_range('angs', args.angs or '0,45,4')
axrats = define_parameter_range('axrats', args.axrats or '0.1,1,4')
res = define_parameter_range('res', args.res or '0,%f,5' % (width/2.,))
boxes = define_parameter_range('boxes', args.boxes or '-0.5,0.5,3')

print("Benchmark measuring profile image of %d x %d with %d iterations" % (width, height, n_iter,))
print("\n%d combinations to be benchmarked" % (len(nsers) * len(angs) * len(axrats) * len(res) * len(boxes)))
print("Parameter ranges: ")

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
print("\nOpenCL platforms/devices information:")
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


parameters = (nsers, angs, axrats, res, boxes)
times = collections.defaultdict(list)
for label_and_evalargs in zip(labels, eval_args):
    label, args = label_and_evalargs
    sys.stdout.write('Evaluating %s...' % label)
    sys.stdout.flush()

    start = time.time()
    for nser, ang, axrat, re, box in itertools.product(*parameters):
        sersic_profile['nser'] = nser
        sersic_profile['ang'] = ang
        sersic_profile['axrat'] = axrat
        sersic_profile['re'] = re
        sersic_profile['box'] = box
        times[label].append(time_me(width=width, height=height, profiles=profiles, **args))

    print(" done! (%.3f [s])" % (time.time() - start))

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