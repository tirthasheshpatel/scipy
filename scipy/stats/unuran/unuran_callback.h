#include <Python.h>
#include <setjmp.h>
#include "ccallback.h"
#include "unuran.h"

#define UNURAN_THUNK(CAST_FUNC, CAST_BACK_FUNC, FUNC, ARG)                          \
    ccallback_t *err_callback = ccallback_obtain();                                 \
    unuran_callback_t *unur_callback = (unuran_callback_t *)(err_callback->info_p); \
    PyObject *extra_arguments = unur_callback->params;                              \
    PyObject *py_function = PyDict_GetItemString(unur_callback->callbacks, FUNC);   \
                                                                                    \
    PyObject *arg1 = NULL, *argobj = NULL, *arglist = NULL, *res = NULL;            \
    double result = 0.;                                                             \
    int error = 0;                                                                  \
                                                                                    \
    argobj = CAST_FUNC(ARG);                                                        \
    if (argobj == NULL) {                                                           \
        error = 1;                                                                  \
        goto done;                                                                  \
    }                                                                               \
                                                                                    \
    arg1 = PyTuple_New(1);                                                          \
    if (arg1 == NULL) {                                                             \
        error = 1;                                                                  \
        goto done;                                                                  \
    }                                                                               \
                                                                                    \
    PyTuple_SET_ITEM(arg1, 0, argobj);                                              \
    argobj = NULL;                                                                  \
                                                                                    \
    arglist = PySequence_Concat(arg1, extra_arguments);                             \
    if (arglist == NULL) {                                                          \
        error = 1;                                                                  \
        goto done;                                                                  \
    }                                                                               \
                                                                                    \
    res = PyObject_CallObject(py_function, arglist);                                \
    if (res == NULL) {                                                              \
        error = 1;                                                                  \
        goto done;                                                                  \
    }                                                                               \
                                                                                    \
    result = CAST_BACK_FUNC(res);                                                   \
    if (PyErr_Occurred()) {                                                         \
        error = 1;                                                                  \
        goto done;                                                                  \
    }                                                                               \
                                                                                    \
done:                                                                               \
    Py_XDECREF(arg1);                                                               \
    Py_XDECREF(argobj);                                                             \
    Py_XDECREF(arglist);                                                            \
    Py_XDECREF(res);                                                                \
                                                                                    \
    if (error) {                                                                    \
        longjmp(err_callback->error_buf, 1);                                        \
    }                                                                               \
                                                                                    \
    return result


typedef struct unuran_callback {
    PyObject *callbacks;
    PyObject *params;
} unuran_callback_t;

static ccallback_signature_t unuran_call_signatures[] = {
    {NULL}
};

int init_unuran_callback(ccallback_t *callback, unuran_callback_t *unur_callback,
                         PyObject *fcn_dict, PyObject *extra_args)
{
    PyObject *fcn;
    int ret;
    int flags = CCALLBACK_OBTAIN;

    callback->c_function = NULL;
    callback->user_data = NULL;
    callback->signature = NULL;

    unur_callback->callbacks = fcn_dict;
    Py_INCREF(fcn_dict);
    unur_callback->params = extra_args;
    Py_INCREF(extra_args);

    /* Store a dummy function to prepare a callback. */
    fcn = PyRun_String("lambda: None", Py_eval_input, PyEval_GetGlobals(), NULL);
    if (fcn == NULL) {
        goto fail;
    }

    ret = ccallback_prepare(callback, unuran_call_signatures, fcn, flags);
    if (ret == -1) {
        goto fail;
    }

    callback->info_p = (void *)unur_callback;

    return 0;

fail:
    Py_DECREF(fcn_dict);
    Py_DECREF(extra_args);
    Py_XDECREF(fcn);
    return -1;
}

int release_unuran_callback(ccallback_t *callback, unuran_callback_t *unur_callback) {
    Py_XDECREF(unur_callback->callbacks);
    Py_XDECREF(unur_callback->params);
    Py_XDECREF(callback->py_function);
    unur_callback->callbacks = NULL;
    unur_callback->params = NULL;
    int ret = ccallback_release(callback);
    return ret;
}

void error_handler(const char *objid, const char *file, int line, const char *errortype,
                   int unur_errno, const char *reason)
{
    ccallback_t *err_callback;
    const char *errno_msg;
    char reason_[256], objid_[256], errno_msg_[256];

    if (unur_errno != UNUR_SUCCESS) {

    err_callback = ccallback_obtain();

    if (reason == NULL || strcmp(reason, "") == 0)
        strcpy(reason_, "unknown error");
    else
        strcpy(reason_, reason);
    if (objid == NULL || strcmp(objid, "") == 0)
        strcpy(objid_, "UNK");
    else
        strcpy(objid_, objid);

    errno_msg = unur_get_strerror(unur_errno);

    if (errno_msg == NULL || strcmp(errno_msg, "") == 0)
        strcpy(errno_msg_, "unknown type");
    else
        strcpy(errno_msg_, errno_msg);
    if (strcmp(errortype, "error") == 0) {
        PyErr_Format(PyExc_ValueError,
                        "[objid: %s] %d : %s => %s",
                        objid_, unur_errno, reason_,
                        errno_msg_);
        longjmp(err_callback->error_buf, 1);
    }
    else  /* warning */
        PyErr_WarnFormat(PyExc_UserWarning, 1,
                            "[objid: %s] %d : %s => %s",
                            objid_, unur_errno, reason_,
                            errno_msg_);

    }
}

double pmf_thunk(int k, const struct unur_distr *distr)
{
    UNURAN_THUNK(PyLong_FromLong, PyFloat_AsDouble, "pmf", k);
}

double pdf_thunk(double x, const struct unur_distr *distr)
{
    UNURAN_THUNK(PyFloat_FromDouble, PyFloat_AsDouble, "pdf", x);
}

double dpdf_thunk(double x, const struct unur_distr *distr)
{
    UNURAN_THUNK(PyFloat_FromDouble, PyFloat_AsDouble, "dpdf", x);
}

double cont_cdf_thunk(double x, const struct unur_distr *distr)
{
    UNURAN_THUNK(PyFloat_FromDouble, PyFloat_AsDouble, "cdf", x);
}

double discr_cdf_thunk(int k, const struct unur_distr *distr)
{
    UNURAN_THUNK(PyLong_FromLong, PyFloat_AsDouble, "cdf", k);
}
