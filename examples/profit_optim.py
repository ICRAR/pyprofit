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
This module defines a function that evaluates how close a generated image is
to a real image for a given set of model parameters. This function can then be
fed into an optimizer which will try to find the best set of parameters that
bring the model close to the image.
"""

import math
import pyprofit

from scipy import signal
from scipy import stats

import numpy as np


class Data(object):
    pass

def to_pyprofit_image(params, data):

    # merge, un-sigma all, un-log some
    sigmas_tofit = data.sigmas[data.tofit]
    allparams = data.model0.copy()
    allparams[data.tofit] = params * sigmas_tofit
    allparams[data.tolog] = 10**allparams[data.tolog]

    fields = ['xcen','ycen','mag','re','nser','ang','axrat','box','calcregion']
    s1params = [x for i,x in enumerate(allparams) if i%2 == 0]
    s1params.append(data.calcregion)
    s2params = [x for i,x in enumerate(allparams) if i%2 != 0]
    s2params.append(data.calcregion)
    if hasattr(data, 'psf'):
        fields.append('convolve')
        s1params.append(True)
        s2params.append(True)

    sparams = [{name: val for name, val in zip(fields, params)} for params in (s1params, s2params)]
    if data.verbose:
        print sparams

    profit_model = {'width':  data.image.shape[1],
                    'height': data.image.shape[0],
                    'magzero': data.magzero,
                    'psf': data.psf,
                    'profiles': {'sersic': sparams}
                   }
    return allparams, np.array(pyprofit.make_model(profit_model))


def profit_like_model(params, data):

    # Get the priors sum
    priorsum = 0
    sigmas_tofit = data.sigmas[data.tofit]
    for i, p in enumerate(data.priors):
        priorsum += p(data.init[i] - params[i]*sigmas_tofit[i])

    # Calculate the new model
    allparams, modelim = to_pyprofit_image(params, data)

    # Scale and stuff
    scaledata = (data.image[data.region] - modelim[data.region])/data.sigim[data.region]
    variance = scaledata.var()
    dof = 2*variance/(variance-1)
    dof = max(min(dof,float('inf')),0)

    ll = np.sum(stats.t.logpdf(scaledata, dof))
    lp = ll + priorsum
    lp = -lp

    if data.verbose:
        print lp, {name: val for name, val in zip(data.names, allparams)}
    return lp

def profit_setup_data(magzero,
                      image, mask, sigim, segim, psf,
                      names, model0, tofit, tolog, sigmas, priors, lowers, uppers):

    im_w, im_h = image.shape
    psf_w, psf_h = psf.shape

    # All the center containing the PSF is considered, as well
    # as the section of the image containing the galaxy
    region = np.zeros(image.shape, dtype=bool)
    region[(im_w - psf_w)/2:(im_w + psf_w)/2][(im_h - psf_h)/2:(im_h + psf_h)/2] = True
    segim_center_pix = segim[int(math.ceil(im_w/2.))][int(math.ceil(im_h/2.))]
    region[segim == segim_center_pix] = True

    # Use the PSF to calculate 'calcregion', which is where we
    # effectively calculate the sersic profile
    psf[psf<0] = 0
    calcregion = signal.convolve2d(region.copy(), psf+1, mode='same')
    calcregion = calcregion > 0

    data = Data()
    data.magzero = magzero
    data.names = names
    data.model0 = model0
    data.tolog = np.logical_and(tolog, tofit)
    data.tofit = tofit
    data.sigmas = sigmas
    data.image = image
    data.sigim = sigim
    data.psf = psf
    data.priors = priors[tofit]
    data.region = region
    data.calcregion = calcregion
    data.verbose = False

    # copy initial parameters
    # log some, /sigma all, filter
    data.init = data.model0.copy()
    data.init[tolog] = np.log10(data.model0[tolog])
    data.init = data.init/sigmas
    data.init = data.init[tofit]

    # Boundaries are scaled by sigma values as well
    data.bounds = np.array(zip(lowers/sigmas,uppers/sigmas))[tofit]

    return data
