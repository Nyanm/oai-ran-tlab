#ifndef OAIPYLIB_WRAPPERS_H
#define OAIPYLIB_WRAPPERS_H

#include <Python.h>

PyObject *py_oaipylib_init(PyObject *self, PyObject *args);
PyObject *py_oaipylib_shutdown(PyObject *self, PyObject *args);
//PyObject *py_oaipylib_nr_polar_encoder(PyObject *self, PyObject *args);
PyObject *py_oaipylib_nr_polar_decoder(PyObject *self, PyObject *args);

PyObject *oaipylib_raise_error(const char *context);
void oaipylib_set_exception_object(PyObject *exc);

#endif
