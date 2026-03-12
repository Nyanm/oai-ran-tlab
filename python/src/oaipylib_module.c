#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "oaipylib_wrappers.h"

static PyMethodDef oaipylib_methods[] = {
    {"init", py_oaipylib_init, METH_NOARGS, "Initialize the underlying C library."},
    {"shutdown", py_oaipylib_shutdown, METH_NOARGS, "Shutdown the underlying C library."},
    {"nr_polar_encoder", py_oaipylib_nr_polar_encoder, METH_VARARGS, "3GPP NR Polar Encoder"},
    {"nr_polar_decoder", py_oaipylib_nr_polar_decoder, METH_VARARGS, "16-bit LLR 3GPP NR Polar Decoder"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef oaipylib_module = {
    PyModuleDef_HEAD_INIT,
    "oaipylib",
    "Python bindings for the OAI C library.",
    -1,
    oaipylib_methods
};

PyMODINIT_FUNC PyInit_oaipylib(void) {
    PyObject *module = PyModule_Create(&oaipylib_module);
    if (!module) {
        return NULL;
    }

    PyObject *error = PyErr_NewException("oaipylib.Error", NULL, NULL);
    if (!error) {
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddObject(module, "Error", error) < 0) {
        Py_DECREF(error);
        Py_DECREF(module);
        return NULL;
    }

    oaipylib_set_exception_object(error);

    return module;
}
