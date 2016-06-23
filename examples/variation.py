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
Example on how to use pyprofit to produce a series of PNG images while varying
one of the sersic parameters. The output can then be piped through ffmpeg/avconv
like this:

python variation.py -p box -f -1 -t 1 -s 0.05 | ffmpeg -i - -f image2pipe -r 5 box-variation.avi
"""

import numpy as np
import png

import pyprofit
import optparse
import sys

width = 500
height = 500

sp = {'xcen': width/2, 'ycen': height/2, 'mag': 15, 'ang': 0, 'box': 0.4, 'axrat': 0.3, 'nser': 4, 're': width/4}
model = {'width': width, 'height': height, 'profiles': {'sersic': [sp]}}

def loggray(x, a=None, b=None):
    a = a or np.min(x)
    b = b or np.max(x)
    linval = 10.0 + 990.0 * (x-float(a))/(b-a)
    return (np.log10(linval)-1.0)*0.5 * 255.0

def variate(param, start, stop, step):
    for b in np.arange(start, stop, step):
        sp[param] = b
        image = np.array(pyprofit.make_model(model))  # @UndefinedVariable
        w = png.Writer(width, height, greyscale=True)
        with open("/dev/stdout", 'w') as f:
            w.write(f, loggray(image))
            f.flush()

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option('-p', action='store', dest='param', help='parameter name')
    parser.add_option('-f', action='store', type='float', dest='start', help='start value (from)')
    parser.add_option('-t', action='store', type='float', dest='stop', help='stop value (to)')
    parser.add_option('-s', action='store', type='float', dest='step', help='step')

    opts, args = parser.parse_args(sys.argv)

    if opts.param is None or opts.start is None or opts.stop is None or opts.step is None:
        parser.error("All arguments are required")

    variate(opts.param, opts.start, opts.stop, opts.step);