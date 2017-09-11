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
import itertools
import random
import time

import pyprofit

parser = argparse.ArgumentParser('')
parser.add_argument('-n', '--niter', help='Number of iterations, defaults to 100',
                    type=int, default=100)

args = parser.parse_args()
n_iter = args.niter

# What we use to time iterative executions
def time_me(**kwargs):
    try:
        start = time.time()
        for _ in range(n_iter):
            pyprofit.make_model(kwargs)
        t = time.time() - start

        return t / n_iter
    except pyprofit.error:
        return float('nan')

# The profile and image/kernel sizes used throughout the benchmark
img_sizes= (100, 150, 200, 300, 400, 800)
krn_sizes = (25, 50, 100, 200)
profiles = {
    'sersic': [
        {'xcen': 50, 'ycen': 50, 'nser': 9.23, 'ang': 23.6, 'axrat': 0.3, 'mag': 10, 'rough': 0, 're': 50}
    ]
}

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
print('Getting an OpenCL environment for each of them now...')

# Get an float OpenCL environment for each of them
# If the device supports double, get a double OpenCL environment as well
openclenvs = []
for p, dev, double_support in all_cl_devs():
    openclenvs.append(pyprofit.openclenv(p, dev, False))
    if double_support:
        openclenvs.append(pyprofit.openclenv(p, dev, True))

# Build up the title and display it
title = "Img Krn    NoConv   Brute    FFT_0    FFT_1"
for plat, dev, has_double_support in all_cl_devs():
    title += "  cl_{0}{1}_f Lcl_{0}{1}_f".format(plat, dev)
    if has_double_support:
        title += "  cl_{0}{1}_d Lcl_{0}{1}_d".format(plat, dev)
print('')
print(title)


# Main benchmarking process
for img_size, krn_size in itertools.product(img_sizes, krn_sizes):

    # We don't test these
    if krn_size > img_size:
        continue

    # Create all required convolvers
    brute_convolver = pyprofit.make_convolver(width=img_size, height=img_size, psf=krns[krn_size])
    fft0_convolver = pyprofit.make_convolver(width=img_size, height=img_size, psf=krns[krn_size], convolver_type='fft', fft_effort=0)
    fft1_convolver = pyprofit.make_convolver(width=img_size, height=img_size, psf=krns[krn_size], convolver_type='fft', fft_effort=1)

    # cl_convolvers include normal and local CL convolvers, in the correct order
    cl_convolvers = []
    for clenv in openclenvs:
        cl_convolvers.append(pyprofit.make_convolver(width=img_size, height=img_size, psf=krns[krn_size], convolver_type='opencl', openclenv=clenv))
        cl_convolvers.append(pyprofit.make_convolver(width=img_size, height=img_size, psf=krns[krn_size], convolver_type='opencl-local', openclenv=clenv))

    # Basic profile calculation time
    profiles['sersic'][0]['convolve'] = False
    t_profile = time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size])
    profiles['sersic'][0]['convolve'] = True

    # ... and convolve with each of them!
    t_brute = time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size], convolver=brute_convolver)
    t_fft0 = time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size], convolver=fft0_convolver)
    t_fft1 = time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size], convolver=fft1_convolver)
    t_cl = []
    for conv in cl_convolvers:
        t_cl.append(time_me(profiles=profiles, width=img_size, height=img_size, psf=krns[krn_size], convolver=conv))

    # Format results and print to the screen
    fmt = "%d %-3d %8.3f %8.3f %8.3f %8.3f"
    fmt += ' ' + ' '.join(['%8.3f'] * (len(t_cl)))
    args = [img_size, krn_size, t_profile, t_brute, t_fft0, t_fft1]
    args += t_cl
    print(fmt % tuple(args))