#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

#include "oaipylib_wrappers.h"
#include "oai_lib_api.h"

static PyObject *OaipylibError = NULL;

void oaipylib_set_exception_object(PyObject *exc) {
    OaipylibError = exc;
}

PyObject *oaipylib_raise_error(const char *context) {
    const char *msg = oai_lib_last_error();
    if (msg == NULL) {
        msg = "unknown library error";
    }

    if (OaipylibError) {
        PyErr_Format(OaipylibError, "%s: %s", context, msg);
    } else {
        PyErr_Format(PyExc_RuntimeError, "%s: %s", context, msg);
    }

    return NULL;
}

PyObject *py_oaipylib_init(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    if (oai_lib_init() != 0) {
        return oaipylib_raise_error("oai_lib_init failed");
    }

    Py_RETURN_NONE;
}

PyObject *py_oaipylib_shutdown(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;

    oai_lib_shutdown();
    Py_RETURN_NONE;
}

PyObject *py_oaipylib_add(PyObject *self, PyObject *args) {
    (void)self;

    double a;
    double b;

    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }

    return PyFloat_FromDouble(oai_lib_add(a, b));
}

PyObject *py_oaipylib_run_algorithm(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *input_obj = NULL;
    double alpha = 0.0;

    if (!PyArg_ParseTuple(args, "Od", &input_obj, &alpha)) {
        return NULL;
    }

    PyObject *seq = PySequence_Fast(input_obj, "input must be a sequence");
    if (!seq) {
        return NULL;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "input sequence must not be empty");
        return NULL;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));

    if (!x || !out) {
        Py_DECREF(seq);
        free(x);
        free(out);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        x[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            free(x);
            free(out);
            return NULL;
        }
    }

    int rc = oai_lib_run_algorithm(x, (int)n, alpha, out);

    Py_DECREF(seq);
    free(x);

    if (rc != 0) {
        free(out);
        return oaipylib_raise_error("oai_lib_run_algorithm failed");
    }

    PyObject *result = PyList_New(n);
    if (!result) {
        free(out);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *value = PyFloat_FromDouble(out[i]);
        if (!value) {
            free(out);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, value);
    }

    free(out);
    return result;
}
