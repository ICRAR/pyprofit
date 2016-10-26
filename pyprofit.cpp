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

#include <sstream>
#include <vector>

#include "gsl/gsl_cdf.h"
#include "gsl/gsl_sf_gamma.h"

#include "profit/brokenexponential.h"
#include "profit/coresersic.h"
#include "profit/ferrer.h"
#include "profit/king.h"
#include "profit/moffat.h"
#include "profit/profit.h"
#include "profit/psf.h"
#include "profit/sersic.h"
#include "profit/sky.h"

using namespace profit;

/* Python 2/3 compatibility */
#if PY_MAJOR_VERSION >= 3
	#define PyInt_AsLong              PyLong_AsLong
	#define PyInt_AsUnsignedLongMask  PyLong_AsUnsignedLongMask
#endif

/* Macros */
#define PYPROFIT_RAISE(str) \
	do { \
		PyErr_SetString(profit_error, str); \
		return NULL; \
	} while (0)

void read_double(PyObject *item, const char *key, double &dst) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		dst = PyFloat_AsDouble(tmp);
	}
}

void read_bool(PyObject *item, const char *key, bool &dst) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		dst = (bool)PyObject_IsTrue(tmp);
	}
}

void read_uint(PyObject *item, const char *key, unsigned int &dst) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		dst = (unsigned int)PyInt_AsUnsignedLongMask(tmp);
	}
}

/* Exceptions */
static PyObject *profit_error;

/* Methods */
static bool *_read_boolean_matrix(PyObject *matrix, unsigned int *matrix_width, unsigned int *matrix_height) {

	bool *bools = NULL;
	Py_ssize_t width = 0, height = 0;

	if( matrix == NULL ) {
		*matrix_width = 0;
		*matrix_height = 0;
		return NULL;
	}

	height = PySequence_Size(matrix);
	for(Py_ssize_t j = 0; j!=height; j++) {
		PyObject *row = PySequence_GetItem(matrix, j);
		if( row == NULL ) {
			free(bools);
			return NULL;
		}

		/* All rows should have the same width */
		if( j == 0 ) {
			width = PySequence_Size(row);
			*matrix_height = (unsigned int)height;
			*matrix_width = (unsigned int)width;
			bools = new bool[*matrix_width * *matrix_height];
		}
		else {
			if( PySequence_Size(row) != width ) {
				Py_DECREF(row);
				free(bools);
				return NULL;
			}
		}

		/* Finally assign the individual values */
		for(Py_ssize_t i=0; i!=width; i++) {
			PyObject *cell = PySequence_GetItem(row, i);
			if( cell == NULL ) {
				Py_DECREF(row);
				free(bools);
				return NULL;
			}
			bools[i + j*width] = (bool)PyObject_IsTrue(cell);
			Py_DECREF(cell);
		}
		Py_DECREF(row);
	}

	return bools;
}

static void _item_to_radial_profile(Profile &profile, PyObject *item) {
	RadialProfile &rp = static_cast<RadialProfile &>(profile);
	read_double(item, "xcen",  rp.xcen);
	read_double(item, "ycen",  rp.ycen);
	read_double(item, "mag",   rp.mag);
	read_double(item, "ang",   rp.ang);
	read_double(item, "axrat", rp.axrat);
	read_double(item, "box",   rp.box);

	read_bool(item, "rough",           rp.rough);
	read_uint(item, "resolution",      rp.resolution);
	read_uint(item, "max_recursions",  rp.max_recursions);
	read_double(item, "acc",           rp.acc);
	read_double(item, "rscale_switch", rp.rscale_switch);

	read_bool(item, "adjust", rp.adjust);
}

static void _item_to_sersic_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	SersicProfile &s = static_cast<SersicProfile &>(profile);
	read_double(item, "re",         s.re);
	read_double(item, "nser",       s.nser);
	read_bool(item, "rescale_flux", s.rescale_flux);
}

static void _item_to_moffat_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	MoffatProfile &m = static_cast<MoffatProfile &>(profile);
	read_double(item, "fwhm",  m.fwhm);
	read_double(item, "con",   m.con);
}

static void _item_to_ferrer_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	FerrerProfile &f = static_cast<FerrerProfile &>(profile);
	read_double(item, "rout", f.rout);
	read_double(item, "a",    f.a);
	read_double(item, "b",    f.b);
}

static void _item_to_coresersic_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	CoreSersicProfile &csp = static_cast<CoreSersicProfile &>(profile);
	read_double(item, "re",   csp.re);
	read_double(item, "rb",   csp.rb);
	read_double(item, "nser", csp.nser);
	read_double(item, "a",    csp.a);
	read_double(item, "b",    csp.b);
}

static void _item_to_brokenexp_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	BrokenExponentialProfile &bep = static_cast<BrokenExponentialProfile &>(profile);
	read_double(item, "h1", bep.h1);
	read_double(item, "h2", bep.h2);
	read_double(item, "rb", bep.rb);
	read_double(item, "a",  bep.a);
}

static void _item_to_king_profile(Profile &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	KingProfile &k = static_cast<KingProfile &>(profile);
	read_double(item, "rc", k.rc);
	read_double(item, "rt", k.rt);
	read_double(item, "a",  k.a);
}

static void _item_to_sky_profile(Profile &profile, PyObject *item) {
	SkyProfile &s = static_cast<SkyProfile &>(profile);
	read_double(item, "bg", s.bg);
}

static void _item_to_psf_profile(Profile &profile, PyObject *item) {
	PsfProfile &psf = static_cast<PsfProfile &>(profile);
	read_double(item, "xcen",  psf.xcen);
	read_double(item, "ycen",  psf.ycen);
	read_double(item, "mag",   psf.mag);
}

void _read_profiles(Model &model, PyObject *profiles_dict, const char *name, void (item_to_profile)(Profile &, PyObject *item)) {

	PyObject *profile_sequence = PyDict_GetItemString(profiles_dict, name);
	if( profile_sequence == NULL ) {
		return;
	}

	Py_ssize_t length = PySequence_Size(profile_sequence);
	for(Py_ssize_t i = 0; i!= length; i++) {
		PyObject *item = PySequence_GetItem(profile_sequence, i);
		try {
			Profile &p = model.add_profile(name);
			read_bool(item, "convolve", p.convolve);
			item_to_profile(p, item);
		} catch(invalid_parameter &e) {
			std::ostringstream os;
			os << "warning: failed to create profile " << name << ": " << e.what();
			PySys_WriteStderr("%s\n", os.str().c_str());
			continue;
		}
		Py_DECREF(item);
	}
}

static void _read_brokenexp_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "brokenexp", &_item_to_brokenexp_profile);
}

static void _read_coresersic_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "coresersic", &_item_to_coresersic_profile);
}

static void _read_ferrer_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "ferrer", &_item_to_ferrer_profile);
	_read_profiles(model, profiles_dict, "ferrers", &_item_to_ferrer_profile);
}

static void _read_king_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "king", &_item_to_king_profile);
}

static void _read_moffat_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "moffat", &_item_to_moffat_profile);
}

static void _read_psf_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "psf", &_item_to_psf_profile);
}

static void _read_sky_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "sky", &_item_to_sky_profile);
}

static void _read_sersic_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "sersic", &_item_to_sersic_profile);
}

static double *_read_psf(PyObject *model_dict, unsigned int *psf_width, unsigned int *psf_height) {

	double *psf = NULL;
	Py_ssize_t width = 0, height = 0;

	PyObject *matrix = PyDict_GetItemString(model_dict, "psf");
	if( matrix == NULL ) {
		*psf_width = 0;
		*psf_height = 0;
		return NULL;
	}

	height = PySequence_Size(matrix);
	for(Py_ssize_t j = 0; j!=height; j++) {
		PyObject *row = PySequence_GetItem(matrix, j);
		if( row == NULL ) {
			free(psf);
			return NULL;
		}

		/* All rows should have the same width */
		if( j == 0 ) {
			width = PySequence_Size(row);
			*psf_height = (unsigned int)height;
			*psf_width = (unsigned int)width;
			psf = new double[width * height];
		}
		else {
			if( PySequence_Size(row) != width ) {
				Py_DECREF(row);
				free(psf);
				return NULL;
			}
		}

		/* Finally assign the individual values */
		for(Py_ssize_t i=0; i!=width; i++) {
			PyObject *cell = PySequence_GetItem(row, i);
			if( cell == NULL ) {
				Py_DECREF(row);
				free(psf);
				return NULL;
			}
			psf[i + j*width] = PyFloat_AsDouble(cell);
			Py_DECREF(cell);
		}
		Py_DECREF(row);
	}

	return psf;
}

#define READ_DOUBLE(from, name, to) \
	do { \
		PyObject *_val = PyDict_GetItemString(from, name); \
		if( _val != NULL ) { \
			to = PyFloat_AsDouble(_val); \
			if( PyErr_Occurred() ) { \
				PYPROFIT_RAISE("Error reading '"#name"' argument, not a floating point number"); \
			} \
		} \
	} while(0);

static PyObject *pyprofit_make_model(PyObject *self, PyObject *args) {

	unsigned int i, j, psf_width = 0, psf_height = 0;
	unsigned int mask_w = 0, mask_h = 0;
	const char *error = NULL;
	double *psf;
	bool *calcmask;

	PyObject *model_dict;
	if( !PyArg_ParseTuple(args, "O!", &PyDict_Type, &model_dict) ) {
		return NULL;
	}

	/* The width, height and profiles are mandatory */
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
	PyObject *profiles_dict = PyDict_GetItemString(model_dict, "profiles");
	if( profiles_dict == NULL ) {
		PYPROFIT_RAISE("Missing mandatory 'profiles' item");
	}

	/* Read the psf if present */
	psf = _read_psf(model_dict, &psf_width, &psf_height);
	if( PyErr_Occurred() ) {
		return NULL;
	}
	calcmask = _read_boolean_matrix(PyDict_GetItemString(model_dict, "calcmask"), &mask_w, &mask_h);
	if( PyErr_Occurred() ) {
		return NULL;
	}
	if( calcmask && (mask_w != width || mask_h != height) ) {
		PYPROFIT_RAISE("calcmask must have same dimensions of image");
	}

	/* Create and initialize the model */
	Model m;
	m.width = width;
	m.height = height;
	READ_DOUBLE(model_dict, "scale_x", m.scale_x);
	READ_DOUBLE(model_dict, "scale_y", m.scale_y);
	if( psf ) {
		m.psf = std::vector<double>(psf, psf + (psf_width * psf_height));
		m.psf_width = psf_width;
		m.psf_height = psf_height;
		READ_DOUBLE(model_dict, "psf_scale_x", m.psf_scale_x);
		READ_DOUBLE(model_dict, "psf_scale_y", m.psf_scale_y);
		delete [] psf;
	}
	if( calcmask ) {
		m.calcmask = std::vector<bool>(calcmask, calcmask + (width * height));
		delete [] calcmask;
	}
	READ_DOUBLE(model_dict, "magzero", m.magzero);

	/* Read the profiles */
	_read_sersic_profiles(m, profiles_dict);
	_read_moffat_profiles(m, profiles_dict);
	_read_ferrer_profiles(m, profiles_dict);
	_read_king_profiles(m, profiles_dict);
	_read_coresersic_profiles(m, profiles_dict);
	_read_brokenexp_profiles(m, profiles_dict);
	_read_sky_profiles(m, profiles_dict);
	_read_psf_profiles(m, profiles_dict);

	/*
	 * Go, Go, Go!
	 * This might take a few [ms], so we release the GIL
	 */
	std::vector<double> image;
	Py_BEGIN_ALLOW_THREADS
	try {
		image = m.evaluate();
	} catch (std::exception &e) {
		error = e.what();
	}
	Py_END_ALLOW_THREADS

	if( error ) {
		PyErr_SetString(profit_error, error);
		return NULL;
	}

	/* Copy resulting image into a 2-D tuple */
	PyObject *image_tuple = PyTuple_New(m.height);
	if( image_tuple == NULL ) {
		PYPROFIT_RAISE("Couldn't create return tuple");
	}

	for(i=0; i!=m.height; i++) {
		PyObject *row_tuple = PyTuple_New(m.width);
		if( row_tuple == NULL ) {
			PYPROFIT_RAISE("Couldn't create row tuple");
		}
		for(j=0; j!=m.width; j++) {
			PyObject *val = PyFloat_FromDouble(image[i*m.width + j]);
			PyTuple_SetItem(row_tuple, j, val);
		}
		PyTuple_SetItem(image_tuple, i, row_tuple);
	}

	/* Clean up and return */
	return image_tuple;
}

static PyMethodDef pyprofit_methods[] = {
    {"make_model",  pyprofit_make_model, METH_VARARGS, "Creates a profit model."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization */

/* Support for Python 2/3 */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "pyprofot", "libprofit wrapper for python", -1, pyprofit_methods};
	#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
	#define MOD_DEF(m, name, doc, methods) \
		m = PyModule_Create(&moduledef);
	#define MOD_VAL(v) v
#else
	#define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
	#define MOD_DEF(m, name, doc, methods) \
		m = Py_InitModule3(name, methods, doc);
	#define MOD_VAL(v)
#endif

extern "C" {

MOD_INIT(pyprofit)
{
	PyObject *m;

	MOD_DEF(m, "pyprofit", "libprofit wrapper for python", pyprofit_methods);
	if( m == NULL ) {
		return MOD_VAL(NULL);
	}

	profit_error = PyErr_NewException((char *)"pyprofit.error", NULL, NULL);
	Py_INCREF(profit_error);
	PyModule_AddObject(m, "error", profit_error);
	return MOD_VAL(m);
}

}
