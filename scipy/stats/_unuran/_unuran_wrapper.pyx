# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np

__all__ = ["randn"]

cdef extern from "unuran.h":
    struct unur_gen
    unur_gen * unur_str2gen(const char *gen_string)
    double unur_sample_cont(unur_gen *rng)
    int unur_sample_vec(unur_gen *rng, double *out)
    void unur_free(unur_gen *rng)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.infer_types(True)   # Infer types whenever possible.
cpdef np.ndarray[np.float64_t, ndim=1] randn(Py_ssize_t size):
    """Generate random numbers from a standard normal distribution."""
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(size, dtype=np.float64)
    cdef unur_gen *rng = unur_str2gen("normal()")
    cdef Py_ssize_t i
    for i in range(size):
        out[i] = unur_sample_cont(rng)
    unur_free(rng)
    return out
