/**
 * Sersic profile implementation
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2016
 * Copyright by UWA (in the framework of the ICRAR)
 * All rights reserved
 *
 * Contributed by Aaron Robotham, Rodrigo Tobar
 *
 * This file is part of libprofit.
 *
 * libprofit is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libprofit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libprofit.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sersic.h"

static inline
double _sersic_for_xy_r(profit_sersic_profile *sp,
                        double x, double y,
                        double r, bool reuse_r) {
	if( sp->box ) {
		double box = sp->box + 2.;
		r = pow( pow(fabs(x), box) + pow(fabs(y), box), 1./box);
	}
	else if( !reuse_r ){
		r = sqrt(x*x + y*y);
	}
	return exp(-sp->_bn*(pow(r/sp->re,1/sp->nser)-1));
}

static inline
void _sersic_translate_rotate(profit_sersic_profile *sp, double x, double y, double *x_ser, double *y_ser) {
	x -= sp->xcen;
	y -= sp->ycen;
	*x_ser = x * sp->_cos_ang + y * sp->_sin_ang;
	*y_ser = (x * sp->_sin_ang - y * sp->_cos_ang) / sp->axrat;
}

static
double _sersic_sumpix(profit_sersic_profile *sp,
                      double x0, double x1, double y0, double y1,
                      unsigned int recur_level) {

	double xbin = (x1-x0) / sp->resolution;
	double ybin = (y1-y0) / sp->resolution;
	double half_xbin = xbin/2.;
	double half_ybin = ybin/2.;
	double total = 0, subval, testval;
	double x , y, x_ser, y_ser;
	unsigned int i, j;

	bool recurse = sp->resolution > 1 && recur_level < sp->max_recursions;

	/* The middle X/Y value is used for each pixel */
	x = x0;
	for(i=0; i < sp->resolution; i++) {
		x += half_xbin;
		y = y0;
		for(j=0; j < sp->resolution; j++) {
			y += half_ybin;

			_sersic_translate_rotate(sp, x, y, &x_ser, &y_ser);
			subval = _sersic_for_xy_r(sp, x_ser, y_ser, 0, false);

			if( recurse ) {
				testval = _sersic_for_xy_r(sp, x_ser, fabs(y_ser) + fabs(ybin/sp->axrat), 0, false);
				if( fabs(testval/subval - 1.0) > sp->acc ) {
					subval = _sersic_sumpix(sp,
					                        x - half_xbin, x + half_xbin,
					                        y - half_ybin, y + half_ybin,
					                        recur_level + 1);
				}
			}

			total += subval;
			y += half_ybin;
		}

		x += half_xbin;
	}

	/* Average and return */
	return total / (sp->resolution * sp->resolution);
}

static
void profit_make_sersic(profit_profile *profile, profit_model *model, double *image) {

	unsigned int i, j;
	double x, y, pixel_val;
	double x_ser, y_ser, r_ser;
	double half_xbin = model->xbin/2.;
	double half_ybin = model->ybin/2.;
	double bin_area = model->xbin * model->ybin;
	profit_sersic_profile *sp = (profit_sersic_profile *)profile;

	/* The middle X/Y value is used for each pixel */
	x = 0;
	for(i=0; i < model->width; i++) {
		x += half_xbin;
		y = 0;
		for(j=0; j < model->height; j++) {
			y += half_ybin;

			_sersic_translate_rotate(sp, x, y, &x_ser, &y_ser);

			/*
			 * No need for further refinement, return sersic profile
			 * TODO: the radius calculation doesn't take into account boxing
			 */
			r_ser = sqrt(x_ser*x_ser + y_ser*y_ser);
			if( sp->rough || sp->nser < 0.5 || r_ser/sp->re > sp->re_switch ){
				pixel_val = _sersic_for_xy_r(sp, x_ser, y_ser, r_ser, true);
			}
			else {
				/* Subsample and integrate */
				pixel_val =  _sersic_sumpix(sp,
				                            x - model->xbin/2, x + model->xbin/2,
				                            y - model->ybin/2, y + model->ybin/2,
				                            0);
			}

			image[i + j*model->width] = bin_area * sp->_ie * pixel_val;
			y += half_ybin;
		}
		x += half_xbin;
	}

}

static
void profit_init_sersic(profit_profile *profile, profit_model *model) {

	profit_sersic_profile *sersic_p = (profit_sersic_profile *)profile;
	double nser = sersic_p->nser;
	double re = sersic_p->re;
	double axrat = sersic_p->axrat;
	double mag = sersic_p->mag;
	double box = sersic_p->box + 2;
	double magzero = model->magzero;
	double bn, angrad, cos_ang;

	if( !sersic_p->_qgamma ) {
		profile->error = strdup("Missing qgamma function on sersic profile");
		return;
	}
	if( !sersic_p->_gammafn ) {
		profile->error = strdup("Missing gamma function on sersic profile");
		return;
	}
	if( !sersic_p->_beta ) {
		profile->error = strdup("Missing beta function on sersic profile");
		return;
	}

	/*
	 * Calculate the total luminosity used by the sersic profile, used
	 * later to calculate the exact contribution of each pixel.
	 * We save bn back into the profile because it's needed later.
	 */
	sersic_p->_bn = bn = sersic_p->_qgamma(0.5, 2*nser, 1);
	double Rbox = M_PI * box / (4*sersic_p->_beta(1/box, 1 + 1/box));
	double gamma = sersic_p->_gammafn(2*nser);
	double lumtot = pow(re, 2) * 2 * M_PI * nser * gamma * axrat/Rbox * exp(bn)/pow(bn, 2*nser);
	sersic_p->_ie = pow(10, -0.4*(mag - magzero))/lumtot;

	/*
	 * Get the rotation angle in radians and calculate the coefficients
	 * that will fill the rotation matrix we'll use later to transform
	 * from image coordinates into sersic coordinates.
	 */
	angrad = fmod(sersic_p->ang, 360.) * M_PI / 180.;
	sersic_p->_cos_ang = cos_ang = cos(angrad);
	sersic_p->_sin_ang = sqrt(1. - cos_ang * cos_ang) * (angrad < M_PI ? -1. : 1.); /* cos^2 + sin^2 = 1 */

}

profit_profile *profit_create_sersic() {
	profit_sersic_profile *p = (profit_sersic_profile *)malloc(sizeof(profit_sersic_profile));
	p->profile.init_profile = &profit_init_sersic;
	p->profile.make_profile = &profit_make_sersic;

	/* Sane defaults */
	p->xcen = 0;
	p->ycen = 0;
	p->mag = 15;
	p->re = 1;
	p->nser = 1;
	p->box = 0;
	p->ang   = 0.0;
	p->axrat = 1.;
	p->rough = false;

	p->acc = 0.1;
	p->re_switch = 1.;
	p->resolution = 9;
	p->max_recursions = 2;

	p->_qgamma = NULL;
	p->_gammafn = NULL;
	p->_beta = NULL;
	return (profit_profile *)p;
}
