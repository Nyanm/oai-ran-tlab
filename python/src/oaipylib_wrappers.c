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


PyObject *py_oaipylib_nr_polar_decoder(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *input_obj = NULL;
    uint64_t out; // output stored
    uint8_t ones_flag;
    int8_t messageType;
    uint16_t messageLength;
    uint8_t aggregation_level;

    if (!PyArg_ParseTuple(args, "ObcHb", &input_obj, &ones_flag,&messageType,&messageLength,&aggregation_level)) {
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

    int16_t *x = (int16_t *)malloc((size_t)n * sizeof(int16_t));
    

    if (!x) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        x[i] = (int16_t)(32767.0*PyFloat_AsDouble(items[i]));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            free(x);
            return NULL;
        }
    }
    int rc = oai_lib_nr_polar_decoder(x, &out, ones_flag, messageType, messageLength,aggregation_level);

    Py_DECREF(seq);
    free(x);

    if (rc != 0) {
        return oaipylib_raise_error("oai_lib_run_algorithm failed");
    }

    PyObject *result = PyList_New(1);
    if (!result) {
        return NULL;
    }
    PyList_SET_ITEM(result, 0, out);

    return result;
}
