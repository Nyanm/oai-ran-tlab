#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "oaipylib_wrappers.h"

static PyMethodDef oaipylib_methods[] = {
    {"init", py_oaipylib_init, METH_NOARGS, "Initialize the underlying C library."},
    {"shutdown", py_oaipylib_shutdown, METH_NOARGS, "Shutdown the underlying C library."},
    {"add", py_oaipylib_add, METH_VARARGS, "Add two floating-point values."},
    {"run_algorithm", py_oaipylib_run_algorithm, METH_VARARGS, "Run an algorithm on a sequence of floats."},
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
