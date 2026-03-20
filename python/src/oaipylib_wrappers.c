#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <limits.h>

#include "oaipylib_wrappers.h"
#include "oai_lib_api.h"

#define ARRAY_LENGTH 27

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



PyObject *py_oaipylib_nr_polar_encoder(PyObject *self, PyObject *args) {
    (void)self;

    uint64_t A;
    uint32_t **out; // output is allocated in OAI API function
    PyObject *out_obj = NULL;
    PyObject *buffer_obj;
    Py_buffer py_A;
    int32_t crcmask;
    uint8_t ones_flag;
    uint8_t messageType;
    uint16_t messageLength;
    uint8_t aggregation_level;

    printf("Parsing input for py_oaipylib_nr_polar_encoder\n");
    // parse the buffer object instead
    if (!PyArg_ParseTuple(args, "OibbHb", &buffer_obj, &crcmask, &ones_flag,&messageType,&messageLength,&aggregation_level)) {
        return NULL;
    }

    // extracting information from the buffer
    if (PyObject_GetBuffer(buffer_obj, &py_A, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT)==-1){
	    return NULL;
    }

    if(py_A.ndim != 1) {
	    PyErr_SetString(PyExc_TypeError, "Encoder input is expected to be 1-D array");
            PyBuffer_Release(&py_A);
	    return NULL;
    }
    //TODO: check types of the items in the array
    // pass the raw buffer to the C function
    out = malloc(sizeof(uint32_t*));
    printf("Calling oai_lib_nr_polar_encoder(0x%x,%p,%x,%d,%d,%d,%d\n",
            A, out, crcmask, ones_flag, (int8_t)messageType, messageLength,aggregation_level);
    int encodedLength = oai_lib_nr_polar_encoder(py_A.buf, (void**)out, crcmask, ones_flag, (int8_t)messageType, messageLength,aggregation_level);
    if (encodedLength <= 0) return oaipylib_raise_error("oai_lib_nr_polar_encoder failed");

         
    int encodedLength_u32 = (encodedLength>>5) + ((encodedLength&31) > 0 ? 1 : 0);
    printf("encoded Length %d (uint32 list size %d), encoded output %d(0x%x) (first 32 bits)\n",encodedLength,encodedLength_u32,(*out)[0],(*out)[0]);
    // once done working with the buffer, release it
    PyBuffer_Release(&py_A);
    PyObject *result = PyList_New(encodedLength_u32);
    if (!result) {
        free(out);
        free(*out);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < encodedLength_u32; ++i) {
    PyObject *value = PyLong_FromUnsignedLong((unsigned long)((*out)[i]));
        if (!value) {
         Py_DECREF(result);
         free(*out);
         free(out);
          return NULL;
	 }
	 PyList_SET_ITEM(result, i, value);
    }
    free(*out);
    free(out);
    return result;
}

PyObject *py_oaipylib_nr_polar_decoder(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *input_obj = NULL;
    uint64_t out; // output stored
    uint8_t ones_flag;
    int8_t messageType;
    uint16_t messageLength;
    uint8_t aggregation_level;

    if (!PyArg_ParseTuple(args, "ObbHb", &input_obj, &ones_flag,&messageType,&messageLength,&aggregation_level)) {
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

    printf("Allocating x for input of size %d\n",n);
    int16_t *x = (int16_t *)malloc((size_t)n * sizeof(int16_t));
    

    if (!x) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }

    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        x[i] = (int16_t)(PyLong_AsLong(items[i]));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            free(x);
            return NULL;
        }
    }
    printf("Calling oai_lib_nr_polar_decoder with ones_flag %d, messageType %d, messageLength %d, aggregation_level %d\n",
		    ones_flag,messageType,messageLength,aggregation_level);
    int rc = oai_lib_nr_polar_decoder(x, &out, ones_flag, messageType, messageLength,aggregation_level);

    Py_DECREF(seq);
    free(x);

    if (rc != 0) {
        return oaipylib_raise_error("oai_lib_run_algorithm failed");
    }

//    printf("output : %x\n",out);
    PyObject *result = PyList_New(1);
    if (!result) {
        return NULL;
    }
    PyList_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong((unsigned long long)out));

    return result;
}
