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
"""
This script shows how to use the profit_optim module to optimize a set of
parameters using pyprofit.
"""

import itertools

import pyfits
from scipy import optimize
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from profit_optim import profit_setup_data, profit_like_model, to_pyprofit_image


def plot_image_comparison(fitsim, modelim, sigmaim, region):
    fig = plt.figure()
    xlist = np.arange(0, fitsim.shape[1])
    ylist = np.arange(0, fitsim.shape[0])
    X, Y = np.meshgrid(xlist, ylist)
    Z = region
    fitsplot = fig.add_subplot(141)
    fitsplot.imshow(fitsim, cmap='gray', norm=mpl.colors.LogNorm())
    fitsplot.contour(X, Y, Z)
    modplot  = fig.add_subplot(142)
    modplot.imshow(modelim, cmap='gray', norm=mpl.colors.LogNorm())
    modplot.contour(X, Y, Z)
    diffplot = fig.add_subplot(143)
    diffplot.imshow(fitsim - modelim, cmap='gray', norm=mpl.colors.LogNorm())
    diffplot.contour(X, Y, Z)
    histplot = fig.add_subplot(144)
    diff = (fitsim[region] - modelim[region])/sigmaim[region]
    histplot.hist(diff[~np.isnan(diff)], bins=100)

def prior_func(s):
    def norm_with_fixed_sigma(x):
        return stats.norm.logpdf(x, 0, s)
    return norm_with_fixed_sigma

# Initial set of parameters
names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('xcen','ycen','mag','re','nser','ang','axrat','box'), ('sersic1','sersic2'))]
model0 = np.array((84.8832, 84.8832, 94.5951, 94.5951, 16.83217, 16.83217, 7.0574, 14.1148, 4.3776, 1.0000, 140.8191, 140.8191, 1.,    0.4891, 0,     0))
tofit  = np.array((True,    False,   True,    False,   True,     True,     True,   True,    True,   False,  True,     True,     False, True,   True,  False))
tolog  = np.array((False,   False,   False,   False,   False,    False,    True,   True,    True,   True,   False,    False,    True,  True,   False, False))
sigmas = np.array((2,       2,       2,       2,       5,        5,        1,      1,       1,      1,      30,       30,       0.3,   0.3,    0.3,   0.3))
lowers = np.array((0,       0,       0,       0,       10,       10,       0,      0,       -1,     -1,     -180,     -180,     -1,    -1,     -1,    -1))
uppers = np.array((1e3,     1e3,     1e3,     1e3,     30,       30,       2,      2,       1.3,    1.3,    360,      360,      0,     0,      1,     1))
priors = np.array([prior_func(s) for s in sigmas])

# Images on which we'll base the optimization
basename = '/home/rtobar/scm/git/ProFit/inst/extdata/G265911'
image = np.array(pyfits.getdata(basename + 'fitim.fits'))
sigim = np.array(pyfits.getdata(basename + 'sigma.fits'))
segim = np.array(pyfits.getdata(basename + 'segim.fits'))
mask  = np.array(pyfits.getdata(basename + 'mskim.fits'))
psf   = np.array(pyfits.getdata(basename + 'psfim.fits'))

def run():

    # Set up the initial structure that will hold all the data
    # needed afterwards
    data = profit_setup_data(0,
                             image, mask, sigim, segim, psf,
                             names, model0, tofit, tolog, sigmas, priors, lowers, uppers)

    # Go, go, go!
    data.verbose = False
    result = optimize.minimize(profit_like_model, data.init, args=(data,), method='L-BFGS-B', bounds=data.bounds, options={'disp':True})

    # Now result.x contains the optimal set of parameters
    # Plot the initial and final set of parameters to see the difference
    _, modelim0 = to_pyprofit_image(data.init, data, use_mask=False)
    _,  modelim  = to_pyprofit_image(result.x, data, use_mask=False)
    plot_image_comparison(data.image, modelim0, data.sigim, data.region)
    plot_image_comparison(data.image, modelim, data.sigim, data.region)
    return result

if __name__ == '__main__':
    run()
