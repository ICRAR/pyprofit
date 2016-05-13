/**
 * libprofit integration into Python
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2016
 * Copyright by UWA (in the framework of the ICRAR)
 * All rights reserved
 *
 * Contributed by Rodrigo Tobar
 *
 * This file is part of pyprofit.
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

#include <Python.h>

#include "gsl/gsl_cdf.h"
#include "gsl/gsl_sf_gamma.h"

#include "profit.h"
#include "sersic.h"
#include "sky.h"

/* Macros */
#define PYPROFIT_RAISE(str) \
	do { \
		PyErr_SetString(profit_error, str); \
		return NULL; \
	} while (0)

#define READ_DOUBLE_INTO(key, dst) \
	val = PyDict_GetItemString(item, key); \
	if( val != NULL ) { \
		dst = PyFloat_AsDouble(val); \
	}

#define READ_SHORT_INTO(key, dst) \
	val = PyDict_GetItemString(item, key); \
	if( val != NULL ) { \
		dst = (short)PyInt_AsLong(val); \
	}

/* Exceptions */
static PyObject *profit_error;

/* Methods */
static void _item_to_sersic_profile(profit_profile *profile, PyObject *item) {
	profit_sersic_profile *s = (profit_sersic_profile *)profile;
	s->_beta = &gsl_sf_beta;
	s->_gammafn = &gsl_sf_gamma;
	s->_qgamma = &gsl_cdf_gamma_Qinv;

	PyObject *val;
	READ_DOUBLE_INTO("xcen",  s->xcen);
	READ_DOUBLE_INTO("ycen",  s->ycen);
	READ_DOUBLE_INTO("mag",   s->mag);
	READ_DOUBLE_INTO("re",    s->re);
	READ_DOUBLE_INTO("nser",  s->nser);
	READ_DOUBLE_INTO("ang",   s->ang);
	READ_DOUBLE_INTO("axrat", s->axrat);
	READ_DOUBLE_INTO("box",   s->box);
	READ_SHORT_INTO("rough",  s->rough);
}

static void _item_to_sky_profile(profit_profile *profile, PyObject *item) {
	profit_sky_profile *s = (profit_sky_profile *)profile;
	PyObject *val;
	READ_DOUBLE_INTO("bg", s->bg);
}

static profit_profile** _read_profiles(PyObject *model_dict, unsigned int *n_profiles, const char *name, void (item_to_profile)(profit_profile *, PyObject *item)) {

	PyObject *profile_sequence = PyDict_GetItemString(model_dict, name);
	if( profile_sequence == NULL ) {
		*n_profiles = 0;
		return NULL;
	}
	Py_ssize_t length = PySequence_Size(profile_sequence);

	*n_profiles = (unsigned int)length;
	profit_profile **profiles = (profit_profile **)malloc(sizeof(profit_profile *) * *n_profiles);
	for(Py_ssize_t i = 0; i!= length; i++) {
		PyObject *item = PySequence_GetItem(profile_sequence, i);
		profit_profile *p = profit_get_profile(name);
		profiles[i] = p;
		item_to_profile(p, item);
		Py_DECREF(item);
	}

	return profiles;
}

static profit_profile** _read_sky_profiles(PyObject *model_dict, unsigned int *n_profiles) {
	return _read_profiles(model_dict, n_profiles, "sky", &_item_to_sky_profile);
}

static profit_profile** _read_sersic_profiles(PyObject *model_dict, unsigned int *n_profiles) {
	return _read_profiles(model_dict, n_profiles, "sersic", &_item_to_sersic_profile);
}

static PyObject *pyprofit_make_model(PyObject *self, PyObject *args) {

	unsigned int i, j;
	char *error;

	PyObject *model_dict;
	if( !PyArg_ParseTuple(args, "O!", &PyDict_Type, &model_dict) ) {
		return NULL;
	}

	/* The width and height are mandatory */
	PyObject *tmp = PyDict_GetItemString(model_dict, "width");
	if( tmp == NULL ) {
		PYPROFIT_RAISE("Missing mandatory 'width' item");
	}
	long width = PyInt_AsLong(tmp);
	if( PyErr_Occurred() ) {
		return NULL;
	}
	tmp = PyDict_GetItemString(model_dict, "height");
	if( tmp == NULL ) {
		PYPROFIT_RAISE("Missing mandatory 'height' item");
	}
	long height = PyInt_AsLong(tmp);
	if( PyErr_Occurred() ) {
		return NULL;
	}

	/* Read the profiles */
	unsigned int n_sersic;
	unsigned int n_sky;
	profit_profile **sersic_profiles = _read_sersic_profiles(model_dict, &n_sersic);
	profit_profile **sky_profiles = _read_sky_profiles(model_dict, &n_sky);

	profit_model *m = (profit_model *)calloc(1, sizeof(profit_model));
	m->error = NULL;
	m->width = width;
	m->height = height;
	m->res_x = width;
	m->res_y = height;

	/* Assign the individual profiles */
	m->n_profiles = n_sersic + n_sky;
	m->profiles = (profit_profile **)malloc(sizeof(profit_profile *) * m->n_profiles);
	for(i=0; i!=n_sersic; i++) {
		m->profiles[i] = sersic_profiles[i];
	}
	for(i=0; i!=n_sky; i++) {
		m->profiles[n_sersic + i] = sky_profiles[i];
	}
	free(sersic_profiles);
	free(sky_profiles);

	PyObject *magzero = PyDict_GetItemString(model_dict, "magzero");
	if( magzero != NULL ) {
		m->magzero = PyFloat_AsDouble(magzero);
		if( PyErr_Occurred() ) {
			profit_cleanup(m);
			return NULL;
		}
	}

	/*
	 * Go, Go, Go!
	 * This might take a few [ms], so we release the GIL
	 */
	Py_BEGIN_ALLOW_THREADS
	profit_make_model(m);
	error = profit_get_error(m);
	Py_END_ALLOW_THREADS

	if( error ) {
		PyErr_SetString(profit_error, m->error);
		profit_cleanup(m);
		return NULL;
	}

	/* Copy resulting image into a 2-D tuple */
	PyObject *image_tuple = PyTuple_New(m->height);
	if( image_tuple == NULL ) {
		PYPROFIT_RAISE("Couldn't create return tuple");
	}

	for(i=0; i!=m->height; i++) {
		PyObject *row_tuple = PyTuple_New(m->width);
		if( row_tuple == NULL ) {
			PYPROFIT_RAISE("Couldn't create row tuple");
		}
		for(j=0; j!=m->width; j++) {
			PyObject *val = PyFloat_FromDouble(m->image[i*m->width + j]);
			PyTuple_SetItem(row_tuple, j, val);
		}
		PyTuple_SetItem(image_tuple, i, row_tuple);
	}

	/* Clean up and return */
	profit_cleanup(m);
	return image_tuple;
}

static PyMethodDef pyprofit_methods[] = {
    {"make_model",  pyprofit_make_model, METH_VARARGS, "Creates a profit model."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization */
PyMODINIT_FUNC
initpyprofit(void)
{
	PyObject *m = Py_InitModule("pyprofit", pyprofit_methods);
	if( m == NULL ) {
		return;
	}

	profit_error = PyErr_NewException("pyprofit.error", NULL, NULL);
	Py_INCREF(profit_error);
	PyModule_AddObject(m, "error", profit_error);
}
