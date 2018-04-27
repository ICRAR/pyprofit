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

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "profit/profit.h"

using namespace profit;

/* Python 2/3 compatibility */
#if PY_MAJOR_VERSION >= 3
	#define PyInt_AsLong               PyLong_AsLong
	#define PyInt_AsUnsignedLongMask   PyLong_AsUnsignedLongMask
	#define STRING_FROM_UTF8(val, len) PyUnicode_FromStringAndSize((const char *)val, len)
#else
	#define STRING_FROM_UTF8(val, len) PyString_FromStringAndSize((const char *)val, len)
#endif

/* Exceptions */
static PyObject *profit_error;

/* Macros */
#define PYPROFIT_RAISE(str) \
	do { \
		PyErr_SetString(profit_error, str); \
		return NULL; \
	} while (0)


/* OpenCL-related methods/object */
static PyObject *pyprofit_opencl_info(PyObject *self, PyObject *args) {

	std::map<int, OpenCL_plat_info> clinfo;
	try {
		clinfo = get_opencl_info();
	} catch (const opencl_error &e) {
		std::ostringstream os;
		os << "Error while getting OpenCL information: " << e.what();
		PYPROFIT_RAISE(os.str().c_str());
	}

	PyObject *p_clinfo = PyList_New(clinfo.size());
	unsigned int plat = 0;
	for(auto platform_info: clinfo) {

		auto plat_info = std::get<1>(platform_info);

		unsigned int dev = 0;
		PyObject *p_devsinfo = PyList_New(plat_info.dev_info.size());
		for(auto device_info: plat_info.dev_info) {

			PyObject *double_support = std::get<1>(device_info).double_support ? Py_True : Py_False;
			Py_INCREF(double_support);

			std::string name = std::move(std::get<1>(device_info).name);
			PyObject *p_name = STRING_FROM_UTF8(name.c_str(), name.size());
			PyObject *p_devinfo = PyTuple_New(2);
			PyTuple_SetItem(p_devinfo, 0, p_name);
			PyTuple_SetItem(p_devinfo, 1, double_support);
			PyList_SetItem(p_devsinfo, dev++, p_devinfo);
		}

		std::string name = std::move(plat_info.name);
		PyObject *p_name = STRING_FROM_UTF8(name.c_str(), name.size());

		PyObject *p_platinfo = PyTuple_New(3);
		PyTuple_SetItem(p_platinfo, 0, p_name);
		PyTuple_SetItem(p_platinfo, 1, PyFloat_FromDouble(plat_info.supported_opencl_version/100.));
		PyTuple_SetItem(p_platinfo, 2, p_devsinfo);
		PyList_SetItem(p_clinfo, plat++, p_platinfo);
	}

	return p_clinfo;
}

/*
 * openclenv object structure
 */
typedef struct {
    PyObject_HEAD
	OpenCLEnvPtr env;
} PyOpenCLEnv;


/*
 * __init__, destructor
 */
static int openclenv_init(PyOpenCLEnv *self, PyObject *args, PyObject *kwargs) {

	unsigned int plat_idx, dev_idx;
	PyObject *use_double_o;

	const char *kwlist[] = {"plat_idx", "dev_idx", "use_double", NULL};
	if( !PyArg_ParseTupleAndKeywords(args, kwargs, "IIO", const_cast<char **>(kwlist),
	                                 &plat_idx, &dev_idx, &use_double_o) ) {
		return -1;
	}

	int use_double = PyObject_IsTrue(use_double_o);
	if( use_double == -1 ) {
		return -1;
	}

	try {
		self->env = get_opencl_environment(plat_idx, dev_idx, static_cast<bool>(use_double), false);
	} catch (const opencl_error &e) {
		std::ostringstream os;
		os << "Error while getting OpenCL information: " << e.what();
		PyErr_SetString(profit_error, os.str().c_str());
		return -1;
	}

	return 0;
}

static void openclenv_dealloc(PyOpenCLEnv *self) {
	self->env.reset();
	Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
 * openclenv object type
 */
static PyTypeObject PyOpenCLEnv_Type = {
#if PY_MAJOR_VERSION >= 3
	PyVarObject_HEAD_INIT(NULL, 0)
#else
	PyObject_HEAD_INIT(NULL)
	0,                             /*ob_size*/
#endif
	"pyprofit.openclenv",          /*tp_name*/
	sizeof(PyOpenCLEnv),             /*tp_basicsize*/
};


/* Utility methods */
void read_double(std::shared_ptr<Profile> &p, PyObject *item, const char *key) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		p->parameter(key, PyFloat_AsDouble(tmp));
	}
}

void read_bool(std::shared_ptr<Profile> &p, PyObject *item, const char *key) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		p->parameter(key, (bool)PyObject_IsTrue(tmp));
	}
}

void read_uint(std::shared_ptr<Profile> &p, PyObject *item, const char *key) {
	PyObject *tmp = PyDict_GetItemString(item, key);
	if( tmp != NULL ) {
		p->parameter(key, (unsigned int)PyInt_AsUnsignedLongMask(tmp));
	}
}

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

static void _item_to_radial_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	read_double(profile, item, "xcen");
	read_double(profile, item, "ycen");
	read_double(profile, item, "mag");
	read_double(profile, item, "ang");
	read_double(profile, item, "axrat");
	read_double(profile, item, "box");

	read_bool(profile, item, "rough");
	read_uint(profile, item, "resolution");
	read_uint(profile, item, "max_recursions");
	read_double(profile, item, "acc");
	read_double(profile, item, "rscale_switch");

	read_bool(profile, item, "adjust");
}

static void _item_to_sersic_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "re");
	read_double(profile, item, "nser");
	read_bool(profile, item, "rescale_flux");
}

static void _item_to_moffat_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "fwhm");
	read_double(profile, item, "con");
}

static void _item_to_ferrer_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "rout");
	read_double(profile, item, "a");
	read_double(profile, item, "b");
}

static void _item_to_coresersic_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "re");
	read_double(profile, item, "rb");
	read_double(profile, item, "nser");
	read_double(profile, item, "a");
	read_double(profile, item, "b");
}

static void _item_to_brokenexp_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "h1");
	read_double(profile, item, "h2");
	read_double(profile, item, "rb");
	read_double(profile, item, "a");
}

static void _item_to_king_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	_item_to_radial_profile(profile, item);
	read_double(profile, item, "rc");
	read_double(profile, item, "rt");
	read_double(profile, item, "a");
}

static void _item_to_sky_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	read_double(profile, item, "bg");
}

static void _item_to_null_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
}

static void _item_to_psf_profile(std::shared_ptr<Profile> &profile, PyObject *item) {
	read_double(profile, item, "xcen");
	read_double(profile, item, "ycen");
	read_double(profile, item, "mag");
}

void _read_profiles(Model &model, PyObject *profiles_dict, const char *name, void (item_to_profile)(std::shared_ptr<Profile> &p, PyObject *item)) {

	PyObject *profile_sequence = PyDict_GetItemString(profiles_dict, name);
	if( profile_sequence == NULL ) {
		return;
	}

	Py_ssize_t length = PySequence_Size(profile_sequence);
	for(Py_ssize_t i = 0; i!= length; i++) {
		PyObject *item = PySequence_GetItem(profile_sequence, i);
		try {
			auto p = model.add_profile(name);
			read_bool(p, item, "convolve");
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

static void _read_null_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "null", &_item_to_null_profile);
}

static void _read_sersic_profiles(Model &model, PyObject *profiles_dict) {
	_read_profiles(model, profiles_dict, "sersic", &_item_to_sersic_profile);
}

static double *_read_psf(PyObject *matrix, unsigned int *psf_width, unsigned int *psf_height) {

	double *psf = NULL;
	Py_ssize_t width = 0, height = 0;

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

static double *_read_psf_from_model(PyObject *model_dict, unsigned int *psf_width, unsigned int *psf_height) {

	PyObject *matrix = PyDict_GetItemString(model_dict, "psf");
	if( matrix == NULL ) {
		*psf_width = 0;
		*psf_height = 0;
		return NULL;
	}

	return _read_psf(matrix, psf_width, psf_height);
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


/*
 * Convolver object structure
 */
typedef struct {
    PyObject_HEAD
    std::shared_ptr<Convolver> convolver;
} PyConvolver;


/*
 * destructor
 */
static void convolverptr_dealloc(PyConvolver *self) {
	self->convolver.reset();
	Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
 * PyConvolver object type
 */
static PyTypeObject PyConvolver_Type = {
#if PY_MAJOR_VERSION >= 3
	PyVarObject_HEAD_INIT(NULL, 0)
#else
	PyObject_HEAD_INIT(NULL)
	0,                             /*ob_size*/
#endif
	"pyprofit.convolver",          /*tp_name*/
	sizeof(PyConvolver),          /*tp_basicsize*/
};


static PyObject *pyprofit_make_convolver(PyObject *self, PyObject *args, PyObject *kwargs) {

	unsigned int psf_width = 0, psf_height = 0, width, height;
	const char *convolver_type = "brute";
	PyObject *psf_p;

	unsigned int omp_threads = 1;
	PyObject *reuse_psf_fft = Py_False;
	unsigned int fft_effort = 0;
	PyObject *p_openclenv = NULL;

	const char * fmt = "IIO|zIOIO:make_convolver";

	const char *kwlist[] = {
	    "width", "height", "psf", "convolver_type",
	    "omp_threads", "reuse_psf_fft", "fft_effort", "openclenv",
	    NULL};

	int res = PyArg_ParseTupleAndKeywords(args, kwargs, fmt, const_cast<char **>(kwlist),
	                                      &width, &height, &psf_p, &convolver_type,
	                                      &omp_threads, &reuse_psf_fft, &fft_effort,
	                                      &p_openclenv);

	if (!res) {
		return NULL;
	}

	/* The width, height and profiles are mandatory */
	_read_psf(psf_p, &psf_width, &psf_height);
	if( PyErr_Occurred() ) {
		return NULL;
	}

	ConvolverCreationPreferences conv_prefs;
	conv_prefs.src_dims = {width, height};
	conv_prefs.krn_dims = {psf_width, psf_height};
	conv_prefs.omp_threads = omp_threads;

	conv_prefs.reuse_krn_fft = static_cast<bool>(PyObject_IsTrue(reuse_psf_fft));
	conv_prefs.effort = effort_t(fft_effort);
	if( p_openclenv != NULL ) {
		if( !PyObject_TypeCheck(p_openclenv, &PyOpenCLEnv_Type) ) {
			PYPROFIT_RAISE("Given openclenv is not of type pyprofit.openclenv");
		}
		PyOpenCLEnv *openclenv = reinterpret_cast<PyOpenCLEnv *>(p_openclenv);
		conv_prefs.opencl_env = openclenv->env;
	}

	PyObject *convolver_ptr = PyObject_CallObject((PyObject *)&PyConvolver_Type, NULL);
	if (!convolver_ptr) {
		PYPROFIT_RAISE("Couldn't allocate memory for new convolver");
	}

	std::string error;
	Py_BEGIN_ALLOW_THREADS
	try {
		((PyConvolver *)convolver_ptr)->convolver = create_convolver(convolver_type, conv_prefs);
	} catch (std::exception &e) {
		// can't PyErr_SetString directly here because we don't have the GIL
		error = e.what();
	}
	Py_END_ALLOW_THREADS

	if (!error.empty()) {
		PYPROFIT_RAISE(error.c_str());
	}

	return convolver_ptr;
}

static PyObject *pyprofit_make_model(PyObject *self, PyObject *args) {

	unsigned int i, j, psf_width = 0, psf_height = 0;
	unsigned int mask_w = 0, mask_h = 0;
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
	psf = _read_psf_from_model(model_dict, &psf_width, &psf_height);
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
	m.set_dimensions({static_cast<unsigned int>(width), static_cast<unsigned int>(height)});
	double scale_x = 1, scale_y = 1;
	READ_DOUBLE(model_dict, "scale_x", scale_x);
	READ_DOUBLE(model_dict, "scale_y", scale_y);
	m.set_image_pixel_scale({scale_x, scale_y});
	if( psf ) {
		auto psf_im = Image(std::vector<double>(psf, psf + (psf_width * psf_height)), psf_width, psf_height);
		m.set_psf(std::move(psf_im));
		double psf_scale_x = 1, psf_scale_y = 1;
		READ_DOUBLE(model_dict, "psf_scale_x", psf_scale_x);
		READ_DOUBLE(model_dict, "psf_scale_y", psf_scale_y);
		m.set_psf_pixel_scale({psf_scale_x, psf_scale_y});
		delete [] psf;
	}
	if( calcmask ) {
		auto mask = Mask(std::vector<bool>(calcmask, calcmask + (width * height)), width, height);
		m.set_mask(std::move(mask));
		delete [] calcmask;
	}
	double magzero = 0;
	READ_DOUBLE(model_dict, "magzero", magzero);
	m.set_magzero(magzero);

	/* Assign the OpenCL environment to the model */
	PyObject *p_openclenv = PyDict_GetItemString(model_dict, "openclenv");
	if( p_openclenv != NULL and p_openclenv != Py_None ) {
		if( !PyObject_TypeCheck(p_openclenv, &PyOpenCLEnv_Type) ) {
			PYPROFIT_RAISE("Given openclenv is not of type pyprofit.openclenv");
		}
		PyOpenCLEnv *openclenv = reinterpret_cast<PyOpenCLEnv *>(p_openclenv);
		m.set_opencl_env(openclenv->env);
	}

	/* Assign requested number of OpenMP threads */
	PyObject *p_omp_threads = PyDict_GetItemString(model_dict, "omp_threads");
	if( p_omp_threads != NULL ) {
		m.set_omp_threads((unsigned int)PyInt_AsUnsignedLongMask(p_omp_threads));
	}

	PyObject *convolver = PyDict_GetItemString(model_dict, "convolver");
	if (convolver) {
		m.set_convolver(((PyConvolver *)convolver)->convolver);
	}

	/* Read the profiles */
	_read_sersic_profiles(m, profiles_dict);
	_read_moffat_profiles(m, profiles_dict);
	_read_ferrer_profiles(m, profiles_dict);
	_read_king_profiles(m, profiles_dict);
	_read_coresersic_profiles(m, profiles_dict);
	_read_brokenexp_profiles(m, profiles_dict);
	_read_sky_profiles(m, profiles_dict);
	_read_null_profiles(m, profiles_dict);
	_read_psf_profiles(m, profiles_dict);

	/*
	 * Go, Go, Go!
	 * This might take a few [ms], so we release the GIL
	 */
	Image image;
	Point offset;
	std::string error;
	Py_BEGIN_ALLOW_THREADS
	try {
		image = m.evaluate(offset);
	} catch (std::exception &e) {
		// can't PyErr_SetString directly here because we don't have the GIL
		error = e.what();
	}
	Py_END_ALLOW_THREADS

	if( !error.empty() ) {
		PyErr_SetString(profit_error, error.c_str());
		return NULL;
	}

	/*
	 * We return a 2-element tuple.
	 * Element 0 is a 2-D tuple with the image values
	 * Element 1 is a 2-element tuple with the offset
	 */
	auto im_dims = image.getDimensions();
	PyObject *image_tuple = PyTuple_New(im_dims.y);
	PyObject *offset_tuple = PyTuple_New(2);
	PyObject *return_tuple = PyTuple_New(2);
	if (image_tuple == NULL || offset_tuple == NULL || return_tuple == NULL) {
		PYPROFIT_RAISE("Couldn't create return tuples");
	}

	/* Copy resulting image into a 2-D tuple */
	for(i=0; i!=im_dims.y; i++) {
		PyObject *row_tuple = PyTuple_New(im_dims.x);
		if( row_tuple == NULL ) {
			PYPROFIT_RAISE("Couldn't create row tuple");
		}
		for(j=0; j!=im_dims.x; j++) {
			PyObject *val = PyFloat_FromDouble(image[i*im_dims.x + j]);
			PyTuple_SetItem(row_tuple, j, val);
		}
		PyTuple_SetItem(image_tuple, i, row_tuple);
	}

	/* Copy offset into another 2-element tuple */
	PyTuple_SetItem(offset_tuple, 0, PyFloat_FromDouble(offset.x));
	PyTuple_SetItem(offset_tuple, 1, PyFloat_FromDouble(offset.y));

	PyTuple_SetItem(return_tuple, 0, image_tuple);
	PyTuple_SetItem(return_tuple, 1, offset_tuple);
	return return_tuple;
}

/*
 * Methods in the pyprofit module
 */
static PyMethodDef pyprofit_methods[] = {
    {"make_model",     pyprofit_make_model,     METH_VARARGS, "Creates a profit model."},
    {"make_convolver", (PyCFunction)pyprofit_make_convolver, METH_VARARGS | METH_KEYWORDS, "Creates a reusable convolver."},
    {"opencl_info",    pyprofit_opencl_info,    METH_NOARGS,  "Gets OpenCL environment information."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module initialization */

/* Support for Python 2/3 */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "pyprofit", "libprofit wrapper for python", -1, pyprofit_methods};
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

	// TODO: add error checking for these
	profit::init();
	Py_AtExit(profit::finish);

	MOD_DEF(m, "pyprofit", "libprofit wrapper for python", pyprofit_methods);
	if( m == NULL ) {
		return MOD_VAL(NULL);
	}

	profit_error = PyErr_NewException((char *)"pyprofit.error", NULL, NULL);
	if( !profit_error ) {
		return MOD_VAL(NULL);
	}

	Py_INCREF(profit_error);
	if( PyModule_AddObject(m, "error", profit_error) == -1 ) {
		return MOD_VAL(NULL);
	}

	PyConvolver_Type.tp_flags = Py_TPFLAGS_DEFAULT;
	PyConvolver_Type.tp_doc = "A model convolver";
	PyConvolver_Type.tp_new = PyType_GenericNew;
	PyConvolver_Type.tp_dealloc = (destructor)convolverptr_dealloc;
	PyConvolver_Type.tp_init = (initproc)NULL;
	if( PyType_Ready(&PyConvolver_Type) < 0 ) {
		return MOD_VAL(NULL);
	}
	Py_INCREF(&PyConvolver_Type);

	if (profit::has_opencl()) {
		PyOpenCLEnv_Type.tp_flags = Py_TPFLAGS_DEFAULT;
		PyOpenCLEnv_Type.tp_doc = "An OpenCL environment";
		PyOpenCLEnv_Type.tp_new = PyType_GenericNew;
		PyOpenCLEnv_Type.tp_dealloc = (destructor)openclenv_dealloc;
		PyOpenCLEnv_Type.tp_init = (initproc)openclenv_init;
		if( PyType_Ready(&PyOpenCLEnv_Type) < 0 ) {
			return MOD_VAL(NULL);
		}
		Py_INCREF(&PyOpenCLEnv_Type);
		PyModule_AddObject(m, "openclenv", (PyObject *)&PyOpenCLEnv_Type);
	}

	return MOD_VAL(m);
}

}
